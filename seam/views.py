# Create your views here.
import os
import cv2
from django.shortcuts import render, redirect
from django.views.generic import ListView, CreateView
from django.urls import reverse_lazy, reverse
from django.http.response import HttpResponse, HttpResponseRedirect
from django.conf import settings
from .forms import PostForm
from .models import Image
import mimetypes
from time import time
import time
import numpy as np
from shutil import rmtree
from .tasks import compute_image_energy
import uuid
import boto3
import json


input_path    = settings.INPUT_PATH
output_path   = settings.OUTPUT_PATH
seams_path    = settings.SEAMS_PATH
video_path    = settings.VIDEO_PATH
video_path_aws= settings.VIDEO_PATH2
local_storage = settings.LOCAL_STORAGE_VAL
all_paths     = [output_path, seams_path, input_path, video_path, video_path_aws]

# print("input_path", input_path)
# MYBASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# input_path2 = os.path.join(MYBASE_DIR, "media/images")
# print("input_path2", input_path2)

# if local_storage:
for each_path in all_paths:
    if os.path.exists(each_path):
        rmtree(each_path)
    os.makedirs(each_path)


# stackoverflow how to upload a file in django 5871730
def index(request):
    
    if request.method == "POST":
        form = PostForm(request.POST, request.FILES)
        if form.is_valid():
            if local_storage:
                myfile = Image(cover = request.FILES['cover'])
                myfile.save()
                request.session['slider_value'] = request.POST['slider_value'] 
                print("value from html***************************", request.POST['slider_value'])
            else:
                # print("video_path_aws", video_path_aws)
                user                                  = uuid.uuid4().hex
                request.session['current_user']  = user
                request.session['input_path']    = f'media/{user}/input'
                request.session['output_path']   = f'media/{user}/output'
                request.session['seams_path']    = f'media/{user}/seams'
                request.session['video_path']    = f'media/{user}/video'
                request.session['video_path_aws']    = os.path.join(video_path_aws, f'{user}')
                request.session['bucket_name']   = settings.BUCKET_NAME
                request.session['file_name']   = request.FILES['cover'].name
                request.session['slider_value'] = request.POST['slider_value'] 


                # 
                s3 = boto3.client("s3")
                s3.upload_fileobj(request.FILES['cover'],
                                    request.session['bucket_name'], 
                                        request.session['input_path'])
            return redirect(reverse('seam_carving'))

    else:
        form = PostForm()
    return render(request, "index.html", {"form": form, 'show_progressbar': json.dumps(False)})


def seam_carving(request): 
    """
    Purpose: Receive user image, process the image via worker task.
    """
    data = {}
    print("yo************", request.session['slider_value'])
    data['slider_value']     = request.session['slider_value']
    if not local_storage:
        data['user']          = request.session['current_user']
        data['input_path']    = request.session['input_path']
        data['output_path']   = request.session['output_path']
        data['seams_path']    = request.session['seams_path']
        data['bucket_name']   = request.session['bucket_name']
        data['file_name']     = request.session['file_name']
    task = compute_image_energy.delay(data)
    return render(request, 'main.html', {'task_id': task.task_id, 'show_progressbar': json.dumps(True)})


def video_file(request):
    """
    Purpose: Create video from collection of processed images.
    """
    print("video_file ****************************************")
    # get image size
    if local_storage:
        filename = os.listdir(input_path)[0]
        seam_images = os.listdir(seams_path)
        num_seam_images = len(seam_images)
        seam_image_0 = seam_images[0]
        image = cv2.imread(os.path.join(seams_path, seam_image_0))

    else:
        key = request.session['input_path']
        bucket_name = request.session['bucket_name']
        filename = os.path.basename(key)
        s3 = boto3.resource('s3')
        image = s3.Bucket(bucket_name).Object(key).get().get('Body').read()
        image = cv2.imdecode(np.asarray(bytearray(image)), cv2.IMREAD_COLOR)

        # create the video path by loading empty object
        path = request.session['video_path']
        s3.Bucket(bucket_name).put_object(Key=path, Body="")
    name = os.path.splitext(filename)[0]
    video_cap = cv2.VideoCapture("vid.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    height, width, ch = image.shape

    if local_storage:
        video = cv2.VideoWriter(os.path.join(video_path, f'vid.mp4'), fourcc, 30, (width, height))

    else:
        # -----------------------incomplete figure out how to upload directly to S3 without storing on disk first
        filename = request.session['video_path_aws']
        aws_path = os.path.join(filename, f'vid.mp4')
        video = cv2.VideoWriter(os.path.join(video_path, f'vid.mp4'), fourcc, 30, (width, height))

    img_canvas =  np.uint8(np.zeros((height, width, ch)))
    adj_width = width
    
    # get images and stitch together to form video
    for i in range(20):
        print("i: ", i)
        # incomplete
        # --------------------------------------------
        if local_storage:
            img = cv2.imread(os.path.join(seams_path, f'seam_image{i}.png'))
        else:
            bucket_name = request.session['bucket_name']
            s3 = boto3.resource('s3')
            key = os.path.join(request.session['seams_path'], f'seam_image{i}.png')
            img = s3.Bucket(bucket_name).Object(key).get().get('Body').read()
            img = cv2.imdecode(np.asarray(bytearray(img)), cv2.IMREAD_COLOR)

            # s3 = boto3.client('s3')
            # s3.download_file(bucket_name, key)


        img_canvas[:,0:adj_width,:] = img
        adj_width -= 1
        video.write(img_canvas)
        img_canvas = np.uint8(np.zeros((height, width, ch)))

    cv2.destroyAllWindows()
    video.release()
    return redirect(reverse("download_file"))


def download_file(request):
    """
    Purpose: Return final (last) processed image result to user
    
    """
    print("download_file ****************************************")
    response = None

    if local_storage:
        # -----------------------------------------
        filename = os.listdir(output_path)[0]
        filepath = os.path.join(output_path, filename)
        with open(filepath, 'rb') as path:
            mime_type, _ = mimetypes.guess_type(filepath)
            response = HttpResponse(path, content_type=mime_type)
            response['Content-Disposition'] = f"attachment; filename={filename}"
        # ------------------------------------------

    else:
        bucket_name = request.session['bucket_name']
        output_path_rem = request.session['output_path']
        myresult = "result.png"
        file_location_s3 = os.path.join(output_path_rem, myresult)
        s3 = boto3.client("s3")
        url = s3.generate_presigned_url(
                                            'get_object', 
                                            Params = { 
                                                        'Bucket': bucket_name, 
                                                        'Key': file_location_s3}, 
                                            ExpiresIn = 600, )
        return HttpResponseRedirect(url)

    form = PostForm()
    return render(request, "index.html", {"form": form})


def syntaxclassbasedview():
    # # Class based view approach
    # class index(CreateView):
    #     """
    #     Purpose: Render the initial home page and provide a form for user to submit an image
    #     """

    #     model = Image
    #     form_class = PostForm
    #     template_name = "index.html"
    #     success_url = reverse_lazy("seam_carving")


    # Class based view approach with some additional processing
    # class index(CreateView):
    #     """
    #     Purpose: Render the initial home page and provide a form for user to submit an image
    #     """

    #     model = Image
    #     form_class = PostForm
    #     template_name = "index.html"
    #     success_url = reverse_lazy("seam_carving")

    #     def form_valid(self, form):
    #     #     # Create unique userid and save necessary 'global' parameters using requests
    #         # if not local_storage:
    #         #     user                                  = uuid.uuid4().hex
    #         #     self.request.session['current_user']  = user
    #         #     self.request.session['input_path']    = f'media/{user}/input'
    #         #     self.request.session['output_path']   = f'media/{user}/output'
    #         #     self.request.session['seams_path']    = f'media/{user}/seams'
    #         #     self.request.session['bucket_name']   = 'seam-project'

    #         #     if self.request.method == "POST":
    #         #         s3 = boto3.client("s3")
    #         #         s3.upload_fileobj(self.request.FILES['cover'],
    #         #                             self.request.session['bucket_name'], 
    #         #                                 self.request.session['input_path'])

    #         return redirect(reverse('seam_carving'))

    # def home(request):
    #     return render(request, 'index.html')
    pass
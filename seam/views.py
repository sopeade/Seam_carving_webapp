# Create your views here.
import os
import cv2
from django.shortcuts import render, redirect
from django.views.generic import ListView, CreateView
from django.urls import reverse_lazy, reverse
from django.http.response import HttpResponse
from .forms import PostForm
from .models import Image
import mimetypes
from time import time
import time
import numpy as np
from numpy.lib.stride_tricks import as_strided, sliding_window_view
from shutil import rmtree
from .tasks import compute_image_energy
# from PIL import Image as pil_image

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
download_path = os.path.join(BASE_DIR, "media/download_image")
seams_path = os.path.join(BASE_DIR, "media/download_seams")
input_path = os.path.join(BASE_DIR, "media/images")
video_path = os.path.join(BASE_DIR, "media/download_video")
all_paths = [download_path, seams_path, input_path, video_path]


class index(CreateView):
    """
    Purpose: Clear database of any images, then Render the initial home page and provide a form for user to submit an image
    """
    # input("pause-here1")
    # myfile = Image.objects.all()
    # print("myfile", myfile, myfile[0], type(myfile[0]))
    # val = cv2.imread(myfile[0])
    # x = pil_image.open(Image.objects.all()[0])
    # print("x", x)
    # input("pause-again1")
    Image.objects.all().delete()
    for each_path in all_paths:
        if os.path.exists(each_path):
            rmtree(each_path)
        os.makedirs(each_path)
    model = Image
    form_class = PostForm
    template_name = "index.html"
    success_url = reverse_lazy("seam_carving")

# def index(request):
#     if request.method == "POST":
#         form = PostForm(request.POST, request.FILES)
#         if form.is_valid():
#             form.save()
            


def seam_carving(request): 
    """
    Purpose: Receive user image, process the image.
    """
    # input("pause-here2")
    # myfile = Image.objects.all()

    # print("myfile", myfile, myfile[0])
    # input("pause-again2")
    task = compute_image_energy.delay()
    return render(request, 'main.html', {'task_id': task.task_id})


def video_file(request):
    """
    Purpose: Create video from collection of processed images.
    """
    print("video_file ****************************************")
    filename = os.listdir(os.path.join(BASE_DIR, 'media/images'))[0]
    name = os.path.splitext(filename)[0]
    ext = os.path.splitext(filename)[1]
    # video_cap = cv2.VideoCapture(f"{name}.mp4")
    video_cap = cv2.VideoCapture("vid.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    seam_images = os.listdir(seams_path)
    num_seam_images = len(seam_images)
    seam_image_0 = seam_images[0]
    height, width, ch = cv2.imread(os.path.join(seams_path, seam_image_0)).shape
    # print("video path is :", os.path.join(video_path, f'{name}.mp4'))
    # video = cv2.VideoWriter(os.path.join(video_path, f'{name}.mp4'), fourcc, 30, (width, height))
    video = cv2.VideoWriter(os.path.join(video_path, f'vid.mp4'), fourcc, 30, (width, height))
    img_canvas =  np.uint8(np.zeros((height, width, ch)))
    adj_width = width
    
    for i in range(num_seam_images):
        print("i: ", i)
        img = cv2.imread(os.path.join(seams_path, f'seam_image{i}{ext}'))
        img_canvas[:,0:adj_width,:] = img
        adj_width -= 1
        video.write(img_canvas)
        img_canvas = np.uint8(np.zeros((height, width, ch)))

    cv2.destroyAllWindows()
    video.release()
    return redirect(reverse("download_file"))
    # return redirect(reverse("index"))
    # return render(request, 'test.html')


def download_file(request):
    """
    Purpose: Return final (last) processed image result to user
    
    """
    print("download_file ****************************************")

    filename = os.listdir(download_path)[0]
    filepath = os.path.join(download_path, filename)
    # Open the file for reading content
    response = None
    with open(filepath, 'rb') as path:
        path = open(filepath, 'rb')
        # Set the mime type
        mime_type, _ = mimetypes.guess_type(filepath)
        # Set the return value of the HttpResponse
        response = HttpResponse(path, content_type=mime_type)
        # Set the HTTP header for sending to browser
        response['Content-Disposition'] = f"attachment; filename={filename}"

    # rmtree(download_path)

    # Delete all stored files
    # if os.path.exists(seams_path):
    #     rmtree(seams_path)
    #     os.makedirs(seams_path)
    # if os.path.exists(download_path):
    #     rmtree(download_path)
    #     os.makedirs(download_path)
    # if os.path.exists(video_path):
    #     rmtree(video_path)
    #     os.makedirs(video_path)
    # image_file_path = os.path.join(input_path, os.listdir(input_path)[0])
    # os.remove(image_file_path)
    return response


# Create your views here.
import os
import io
import cv2
from django.shortcuts import render, redirect
from django.views.generic import ListView, CreateView
from django.urls import reverse_lazy, reverse
from django.http.response import HttpResponse, HttpResponseRedirect
from django.conf import settings
from .forms import PostForm
from .models import Image
import mimetypes
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
video_path_aws= settings.VIDEO_PATH_AWS
local_storage = settings.LOCAL_STORAGE_VAL
store_aws_local = settings.STORE_AWS_LOCAL
all_paths     = [output_path, seams_path, input_path, video_path]

if local_storage:
    for each_path in all_paths:
        if os.path.exists(each_path):
            rmtree(each_path)
        os.makedirs(each_path)


# stackoverflow how to upload a file in django 5871730
def index(request):
    '''Get image from user and save image'''
    if request.method == "POST":
        form = PostForm(request.POST, request.FILES)
        if form.is_valid():
            if local_storage:
                myfile = Image(cover = request.FILES['cover'])
                myfile.save()
                request.session['slider_value'] = request.POST['slider_value'] 
                print("value from html***************************", request.POST['slider_value'])
            else:
                user                             = uuid.uuid4().hex
                request.session['current_user']  = user
                request.session['input_path']    = f'media/{user}/input'
                request.session['output_path']   = f'media/{user}/output'
                request.session['seams_path']    = f'media/{user}/seams'
                request.session['video_path']    = f'media/{user}/video'
                request.session['bucket_name']   = settings.BUCKET_NAME
                request.session['file_name']     = request.FILES['cover'].name
                request.session['slider_value']  = request.POST['slider_value']
                 

                path = os.path.join(video_path_aws, f'{user}')
                request.session['user_directory']  = path
                subfolder = ['input_path', 'output_path', 'seams_path', 'video_path']
                if os.path.exists(path):
                    rmtree(path)
                for folder in subfolder:
                    ondrive_path = request.session[folder]
                    os.makedirs(ondrive_path)
                s3 = boto3.client("s3")
                s3.upload_fileobj(request.FILES['cover'],
                                    request.session['bucket_name'], 
                                        request.session['input_path'])
            return redirect(reverse('seam_carving'))

    else:
        form = PostForm()
    return render(request, "index.html", {"form": form, 'show_progressbar': json.dumps(False), 'task_id': ""})


def seam_carving(request): 
    """
    Purpose: Receive user image, process the image via worker task.
    """
    data = {}
    data['slider_value']     = request.session['slider_value']
    if not local_storage:
        data['user']         = request.session['current_user']
        data['input_path']   = request.session['input_path']
        data['output_path']  = request.session['output_path']
        data['seams_path']   = request.session['seams_path']
        data['bucket_name']  = request.session['bucket_name']
        data['file_name']    = request.session['file_name']
        data['video_path']    = request.session['video_path']

    task = compute_image_energy.delay(data)
    return render(request, 'index.html', {'task_id': task.task_id, 'show_progressbar': json.dumps(True)})


def download_image(request):
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
        # return response

    else:
        if store_aws_local:
            filename = os.listdir(request.session['output_path'])[0]
            filepath = os.path.join(request.session['output_path'], filename)
            with open(filepath, 'rb') as path:
                mime_type, _ = mimetypes.guess_type(filepath)
                response = HttpResponse(path, content_type=mime_type)
                response['Content-Disposition'] = f"attachment; filename={filename}"

        else:
        
            bucket_name = request.session['bucket_name']
            output_path_rem = request.session['output_path']
            myresult = "result.png"
            file_location_s3 = os.path.join(output_path_rem, myresult)
            file_obj = download_bytes_object(bucket_name, file_location_s3)
            print("file_obj", type(file_obj))
            mime_type, _ = mimetypes.guess_type(myresult)
            response = HttpResponse(file_obj, content_type=mime_type)
            response['Content-Disposition'] = f"attachment; filename={myresult}"
            # -------------------------------------------------------------------

            # s3 = boto3.client("s3")
            # url = s3.generate_presigned_url(
            #                                     'get_object', 
            #                                     Params = { 
            #                                                 'Bucket': bucket_name, 
            #                                                 'Key': file_location_s3}, 
            #                                     ExpiresIn = 600, )
            # return HttpResponseRedirect(url)
            # print("url", url, type(url))
        
    return response

    # form = PostForm()
    # return render(request, "index.html", {"form": form})


def download_video(request):
    """
    Purpose: Return final (last) processed image result to user
    
    """
    print("download_video ****************************************")
    response = None

    if local_storage:
        # -----------------------------------------
        filename = os.listdir(video_path)[0]
        filepath = os.path.join(video_path, filename)
        with open(filepath, 'rb') as path:
            mime_type, _ = mimetypes.guess_type(filepath)
            response = HttpResponse(path, content_type=mime_type)
            response['Content-Disposition'] = f"attachment; filename={filename}"
        # ------------------------------------------
        # return response

    else:
        if store_aws_local:
            filename = os.listdir(request.session['video_path'])[0]
            filepath = os.path.join(request.session['video_path'], filename)
            with open(filepath, 'rb') as path:
                mime_type, _ = mimetypes.guess_type(filepath)
                response = HttpResponse(path, content_type=mime_type)
                response['Content-Disposition'] = f"attachment; filename={filename}"
                # if os.path.exists(request.session['user_directory']):
                #     rmtree(each_path)
            # return response
        else:
        
            bucket_name = request.session['bucket_name']
            vid_path = request.session['video_path']
            myresult = "result.mp4"
            # file_location_s3 = os.path.join(vid_path, myresult)
            # file_obj = download_bytes_object(bucket_name, file_location_s3)
            file_obj = download_bytes_object(bucket_name, vid_path)
            mime_type, _ = mimetypes.guess_type(myresult)
            response = HttpResponse(file_obj, content_type=mime_type)
            response['Content-Disposition'] = f"attachment; filename={myresult}"
            # ---------------------------------------------------------
            # s3 = boto3.client("s3")
            # url = s3.generate_presigned_url(
            #                                     'get_object', 
            #                                     Params = { 
            #                                                 'Bucket': bucket_name, 
            #                                                 'Key': file_location_s3}, 
            #                                     ExpiresIn = 600, )
            # return url
    return response
    # form = PostForm()
    # return render(request, "index.html", {"form": form})


def download_bytes_object(bucket_name, s3_key):
    s3 = boto3.client("s3")
    obj = s3.get_object(Bucket=bucket_name, Key=s3_key)
    s3_obj = obj.get('Body')
    bytes_object = s3_obj.read()
    return bytes_object
from celery import shared_task
from celery_progress.backend import ProgressRecorder
import os
import cv2
from numpy.lib.stride_tricks import as_strided, sliding_window_view
from django.conf import settings
import numpy as np
from .models import Image
import boto3
import io
import tempfile

input_path_local  = settings.INPUT_PATH
output_path_local = settings.OUTPUT_PATH
seams_path_local  = settings.SEAMS_PATH
video_path_local  = settings.VIDEO_PATH
local_storage = settings.LOCAL_STORAGE_VAL
store_aws_local = settings.STORE_AWS_LOCAL

@shared_task(bind=True)
def compute_image_energy(self, data):
    """
    Purpose: Use provided user image to create and store collection of processed images.
    """
    progress_recorder = ProgressRecorder(self)
    
    if local_storage:
        filename = os.listdir(input_path_local)[0]
        ext = os.path.splitext(filename)[1]
        image = cv2.imread(os.path.join(input_path_local, f'{filename}'))
        slider_value       = data['slider_value']
    else:
        user              = data['user']
        input_path_rem    = data['input_path']
        output_path_rem   = data['output_path']
        seams_path_rem    = data['seams_path']
        video_path_rem    = data['video_path']
        bucket_name       = data['bucket_name']
        slider_value      = data['slider_value']
        filename          = data['file_name']
        ext = os.path.splitext(filename)[1]
        s3 = boto3.resource('s3')
        image = s3.Bucket(bucket_name).Object(input_path_rem).get().get('Body').read()
        image = cv2.imdecode(np.asarray(bytearray(image)), cv2.IMREAD_COLOR)
    num_rows, num_cols, num_chan = image.shape
    pct_seams_to_remove= int(slider_value)
    redSeams = True
    test_array = np.copy(image)
    ini_img = np.copy(image)
    count = 0
    vectorize = True
    mode = 'edge'
    start = len(test_array[0])
    end = int((100-pct_seams_to_remove)/100*num_cols)

    # loop over image, identifying and removing min energy seams
    while len(test_array[0]) > end:
        
        dx = cv2.Sobel(test_array,cv2.CV_64F,1,0,ksize=1)
        dy = cv2.Sobel(test_array,cv2.CV_64F,0,1,ksize=1)
        energy = abs(dy) + abs(dx)
        enrg_r, enrg_c, *_ = energy.shape
        curr_eng = np.sum(energy, axis=2)
        tot_eng = np.copy(curr_eng)
        mask = np.ones_like(tot_eng)
        prev_index = np.zeros_like(tot_eng)
        kernel = np.array([1, 1, 1])
        pad_tot_eng = np.pad(tot_eng, pad_width=((0, 0), (1, 1)), mode=mode)
        pad_curr_eng = np.pad(curr_eng, pad_width=((0, 0), (1, 1)), mode=mode)

        if vectorize:
            for idx_r, _ in enumerate(pad_tot_eng[1:], start=1):
                upper_row = pad_tot_eng[idx_r - 1]
                slide_row = sliding_window_view(upper_row, kernel.shape)   #get a strided view
                min_top_val = np.min(slide_row, axis=1)                    #get the value of the min of each grouping of 3
                min_idx = np.argmin(slide_row, axis=1)                     #get the "pseudo column index" of that minimum
                min_idy = np.arange(len(slide_row))                        #get the "pseudo row index" of that minimum
                min_ind = min_idx + min_idy - 1                            #The actual column index is given by this formula
                if min_ind[0] < 0:                                         
                    min_ind[0] += 1                                        #This is a fix to clip the first value at 0 (currently at -1)
                                                                
                prev_index[idx_r] = min_ind
                curr_row = pad_curr_eng[idx_r]
                curr_row = sliding_window_view(curr_row, kernel.shape)     #strided view of the padded energy of current row
                curr_row = curr_row[:, 1]                                  #middle value (of kernel size 3) is current cell
                total = min_top_val + curr_row
                padded_total = np.pad(total, pad_width=1, mode=mode)
                pad_tot_eng[idx_r] = padded_total
            tot_eng = pad_tot_eng[:,1:-1]
        
        else:
            for idx_r in range(1, len(tot_eng)):
                for idx_c in range(len(tot_eng[0])):
                    if idx_c == 0:
                        tot_eng[idx_r][idx_c] = curr_eng[idx_r][idx_c] + min(tot_eng[idx_r-1][idx_c], tot_eng[idx_r-1][idx_c+1])
                        prev_index[idx_r][idx_c] = idx_c if tot_eng[idx_r-1][idx_c] < tot_eng[idx_r-1][idx_c + 1] else idx_c + 1
            
                    elif idx_c == len(tot_eng[0])-1:
                        tot_eng[idx_r][idx_c] = curr_eng[idx_r][idx_c] + min(tot_eng[idx_r-1][idx_c-1], tot_eng[idx_r-1][idx_c])
                        prev_index[idx_r][idx_c] = idx_c-1 if tot_eng[idx_r-1][idx_c-1] < tot_eng[idx_r-1][idx_c] else idx_c
            
                    else:
                        min_top_col = min(tot_eng[idx_r-1][idx_c-1], tot_eng[idx_r-1][idx_c], tot_eng[idx_r-1][idx_c+1])
                        tot_eng[idx_r][idx_c] = curr_eng[idx_r][idx_c] + min_top_col
            
                        prev_index[idx_r][idx_c] = (idx_c - 1) if (min_top_col == tot_eng[idx_r-1][idx_c-1]) else \
                                                idx_c if (min_top_col == tot_eng[idx_r-1][idx_c]) else \
                                                idx_c + 1

        #  Create mask with False values representing seam
        for index, row_array in enumerate(tot_eng[::-1]):
            row_idx = (len(tot_eng)-1-index)
            if index == 0:
                # Starting from the bottom we find the min column and set the mask to 0
                min_col_idy = np.where(row_array == min(row_array))[0][0]

            mask[row_idx][min_col_idy] = False #this will make mask an array of ones with a strip of zeros
            min_col_idy = int(prev_index[row_idx][min_col_idy]) #get the index of the minimum cell that led to (i.e. above) this cell

        # weight mask against test_array and tot_eng thus removing pixels and reshape array
        test_array = test_array + .1                   # add a small delta to ensure that zeros in the original image are not affected
        test_array_ini = test_array
        test_array = test_array*np.atleast_3d(mask)    #make mask same dimesion as test_array and multiply to get a strip of zeros

        if redSeams:
            boolean_seam = (np.atleast_3d(mask)*-1) + 1    # seam is now a single positive jagged colum of ones
            colored_seam = (boolean_seam * test_array_ini)
            red_seam_2d = colored_seam[:,:,2]
            red_seam_3d = np.zeros(test_array.shape)
            red_seam_3d[:, : ,2] = red_seam_2d
            red_seam_img = test_array + red_seam_3d
            red_seam_img.astype(np.uint8)

            if local_storage:
                cv2.imwrite(os.path.join(seams_path_local, f"seam_image{count}.png"), red_seam_img)
            else:
                if store_aws_local:
                    # cv2.imwrite(os.path.join(seams_aws, f"seam_image{count}.png"), red_seam_img)
                    cv2.imwrite(os.path.join(seams_path_rem, f"seam_image{count}.png"), red_seam_img)
                else:
                    path = os.path.join(seams_path_rem, f'seam_image{count}.png')
                    red_seam_img = cv2.imencode('.png', red_seam_img)[1].tobytes()  # imencode returns success, value so index 1 to obtain value
                    image = s3.Bucket(bucket_name).put_object(Key=path, Body=red_seam_img, ContentType='image/png')

        test_array = (test_array[test_array != 0]).reshape(len(test_array),len(test_array[0])-1,3) # remove blank strip from image and reshape
        test_array = test_array - .1              # remove small delta that was previously added

        # As the image reduced in size (above), we have to reduce the size of the corresponding tot_eng array to match
        tot_eng = tot_eng +.1
        tot_eng = tot_eng*mask
        tot_eng = (tot_eng[tot_eng != 0]).reshape(len(tot_eng),len(tot_eng[0])-1)
        tot_eng = tot_eng - .1
        min_col_idy = np.where(tot_eng[-1] == min(tot_eng[-1]))[0][0]
        count += 1
        begin_pct = int((start - len(test_array[0])) * 100/ (start - end))
        end_pct = 100
        progress_recorder.set_progress(begin_pct, 100)

    test_array.astype(np.uint8)

    # store the final image result
    if local_storage:
        cv2.imwrite(os.path.join(output_path_local, f"result.png"), test_array)
    else:
        if store_aws_local:
            cv2.imwrite(os.path.join(output_path_rem, f"result.png"), test_array)
        else:
            path = os.path.join(output_path_rem, f'result.png')
            result = cv2.imencode('.png', test_array)[1].tobytes()
            s3.Bucket(bucket_name).put_object(Key=path, Body=result, ContentType='image/png')
    
    # create and store the video result
    if local_storage:
        filename = os.listdir(input_path_local)[0]
        seam_images = os.listdir(seams_path_local)
        num_seam_images = len(seam_images)
        seam_image_0 = seam_images[0]
        image = cv2.imread(os.path.join(seams_path_local, seam_image_0))
        name = os.path.splitext(filename)[0]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        height, width, ch = image.shape
        video = cv2.VideoWriter(os.path.join(video_path_local, f'vid.mp4'), fourcc, 30, (width, height))
        img_canvas =  np.uint8(np.zeros((height, width, ch)))
        adj_width = width
        
        # get images and stitch together to form video
        for i in range(count):
            img = cv2.imread(os.path.join(seams_path_local, f'seam_image{i}.png'))


            img_canvas[:,0:adj_width,:] = img
            adj_width -= 1
            video.write(img_canvas)
            img_canvas = np.uint8(np.zeros((height, width, ch)))

        cv2.destroyAllWindows()
        video.release()


    else:
        key = input_path_rem
        s3 = boto3.resource('s3')
        image = s3.Bucket(bucket_name).Object(key).get().get('Body').read()
        image = cv2.imdecode(np.asarray(bytearray(image)), cv2.IMREAD_COLOR)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        height, width, ch = image.shape
        

        # -save video to temp location for situations which do not support saving to drive/filesystem e.g. heroku deploy
        video_file = None
        with tempfile.NamedTemporaryFile() as temp:
            video = cv2.VideoWriter(os.path.join(temp.name, f'vid.mp4'), fourcc, 30, (width, height))
            video_file = temp

        # -------------------------------------------------
        # video = cv2.VideoWriter(os.path.join(video_path_rem, f'vid.mp4'), fourcc, 30, (width, height))
            img_canvas =  np.uint8(np.zeros((height, width, ch)))
            adj_width = width
            
            # get images and stitch together to form video
            for i in range(count):
                if store_aws_local:
                    img = cv2.imread(os.path.join(seams_path_rem, f'seam_image{i}.png'))
                    img_canvas[:,0:adj_width,:] = img
                    adj_width -= 1
                    video.write(img_canvas)
                    img_canvas = np.uint8(np.zeros((height, width, ch)))
                else:
                    s3 = boto3.resource('s3')
                    key = os.path.join(seams_path_rem, f'seam_image{i}.png')
                    img = s3.Bucket(bucket_name).Object(key).get().get('Body').read()
                    img = cv2.imdecode(np.asarray(bytearray(img)), cv2.IMREAD_COLOR)
                    img_canvas[:,0:adj_width,:] = img
                    adj_width -= 1
                    video.write(img_canvas)
                    img_canvas = np.uint8(np.zeros((height, width, ch)))

            cv2.destroyAllWindows()
            video.release()
            print("video_file", video_file, type(video_file), type(video_file.name))
            s3 = boto3.client("s3")
            s3.upload_fileobj(video_file,
                                bucket_name, 
                                    video_path_rem)
    return 'done'
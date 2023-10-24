from celery import shared_task
from celery_progress.backend import ProgressRecorder
from time import sleep
from time import time
import os
import cv2
from numpy.lib.stride_tricks import as_strided, sliding_window_view
import numpy as np


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
download_path = os.path.join(BASE_DIR, "media/download_image")
seams_path = os.path.join(BASE_DIR, "media/download_seams")
input_path = os.path.join(BASE_DIR, "media/images")

@shared_task(bind=True)
def compute_image_energy(self):
    progress_recorder = ProgressRecorder(self)
    filename = os.listdir(os.path.join(BASE_DIR, 'media/images'))[0]
    ext = os.path.splitext(filename)[1]
    image = cv2.imread(os.path.join(BASE_DIR, f'media/images/{filename}'))
    num_rows, num_cols, num_chan = image.shape
    # if (num_rows * num_cols) > 4*1024*1024:
    #     image_file_path = os.path.join(input_path, os.listdir(input_path)[0])
    #     os.remove(image_file_path)
    #     raise ValidationError("Image size too large. Needs to be < 4MB")
    pct_seams_to_remove=0.3
    redSeams = True
    # start = time()
    test_array = np.copy(image)
    ini_img = np.copy(image)
    count = 0
    start_while = time()
    vectorize = True
    mode = 'edge'
    start = len(test_array[0])
    end = int((1-pct_seams_to_remove)*len(image[0]))

    while len(test_array[0]) > int((1-pct_seams_to_remove)*len(image[0])):
        
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
            # vectorize rows (total_time 25secs)
        # def compute_image_energy(pad_tot_eng, pad_curr_eng, kernel, prev_index) 

        # task = compute_image_energy.delay(self, pad_tot_eng, pad_curr_eng, kernel, prev_index)

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
        test_array = test_array + .1
        test_array_ini = test_array
        test_array = test_array*np.atleast_3d(mask)    #strip of zeros

        if redSeams:
            boolean_seam = (np.atleast_3d(mask)*-1) + 1    # seam is now a single positive jagged colum of ones
            colored_seam = (boolean_seam * test_array_ini)
            red_seam_2d = colored_seam[:,:,2]
            red_seam_3d = np.zeros(test_array.shape)
            red_seam_3d[:, : ,2] = red_seam_2d
            red_seam_img = test_array + red_seam_3d
            red_seam_img.astype(np.uint8)
            cv2.imwrite(os.path.join(seams_path, f"seam_image{count}{ext}"), red_seam_img)

        test_array = (test_array[test_array != 0]).reshape(len(test_array),len(test_array[0])-1,3) # remove blank strip from image and reshape
        test_array = test_array - .1

        # As the image reduced in size (above), we have to reduce the size of the corresponding tot_eng array to match
        tot_eng = tot_eng +.1
        tot_eng = tot_eng*mask
        tot_eng = (tot_eng[tot_eng != 0]).reshape(len(tot_eng),len(tot_eng[0])-1)
        tot_eng = tot_eng - .1
        min_col_idy = np.where(tot_eng[-1] == min(tot_eng[-1]))[0][0]
        count += 1
        # print(f"count{count}")
        begin_pct = int((start - len(test_array[0])) * 100/ (start - end))
        end_pct = 100
        progress_recorder.set_progress(begin_pct, 100)
    # end_while = time()

    result = test_array.astype(np.uint8)
    # end = time()
    # total_time = end - start
    # while_time = end_while - start_while

    # print(f"total_time:  {total_time}, time in while loop: {while_time}")
    cv2.imwrite(os.path.join(download_path, f"seam_image{ext}"), result)
    # return redirect(reverse("video_file"))
    return 'Done'
    # return HttpResponse("This was a Success.")


# @shared_task(bind=True)
# def compute_image_energy2(self, seconds):
#     progress_recorder = ProgressRecorder(self)
#     x = 'happy'
#     y = 'sad'
#     for i in range(3):
#         sleep(seconds)
#         progress_recorder.set_progress(i + 1, 3, f'On iteration {i}')
#     return x, y

from cProfile import label
import os
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.python.ops.array_ops import required_space_to_batch_paddings
from tqdm import tqdm
import shutil

from urllib3 import Retry

from ssdforselect import SSD

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)




def get_filelist(path):
    filelist = []
    for root, dirs, files in os.walk(path):
        for file_name in files:
            filelist.append(os.path.join(root,file_name))
    return filelist

def get_info(result):
    # if len(result) == 1:
    #     label = result[0][0]
    #     conf_score = result[0][1]
    #     top = result[0][2]
    #     left = result[0][3]
    #     bottom = result[0][4] 
    #     right = result[0][5]

    #     area = (bottom - top) * (right - left)
    
    #     # return [label, conf_score, area]
    #     return conf_score
    info = {0.0:0, 1.0:0}
    for i in range(len(result)):
        label = result[i][0]
        conf_score = result[i][1]

        if info[label] < conf_score:
            info[label] = conf_score
    
    score = info[0.0] + info[1.0]

        
    return score


if __name__ == "__main__":
    ssd = SSD()

    dir_path = './test_images'
    sa_path = './test_results_test_two'

    dirlist = os.listdir(dir_path)
 
    for dir in dirlist:
        filelist = get_filelist(os.path.join(dir_path, dir))
        file_result_dict = {}
        for file in tqdm(filelist):
            # img = input('Input image filename:')
            try:
                image = Image.open(file)
            except:
                print('Open Error! Try again!')
                continue
            else:
                r_image, r_results = ssd.detect_image(image)
                # r_image.show()
                if len(r_results) == 0:
                    pass
                else:
                    result_info = get_info(r_results)
                    file_result_dict[file] = result_info


        # file_result_dict_items = sorted(file_result_dict.items(), key = lambda x : x[1][1], reverse=True)
        file_result_dict_items = sorted(file_result_dict.items(), key = lambda x : x[1], reverse=True)

        for file_result in file_result_dict_items[:2]:
            id = os.path.split(os.path.split(file_result[0])[0])[1]
            path = os.path.join(sa_path, id)
            if not os.path.exists(path):
                os.makedirs(path)

            shutil.copy(file_result[0], path + os.sep + os.path.split(file_result[0])[1])        


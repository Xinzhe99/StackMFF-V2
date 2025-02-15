# -*- coding: utf-8 -*-
# @Author  : XinZhe Xie
# @University  : ZheJiang University
# -*- coding: utf-8 -*-
# @Author  : XinZhe Xie
# @University  : ZheJiang University

import numpy as np
import math
import os
import skimage
from PIL import Image
import pandas as pd
import cv2
from simple_metric import *
import glob
import re
from natsort import natsorted
from tqdm import tqdm
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from VIF import vifp_mscale
from MI import MI_function
from niqe import niqe
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')
#Parameters to set todo
base_path=r'C:\Users\dell\Desktop\Working\U3D-Fusion\major_revision_self\exp\all_exp'
ground_truth_path=r'E:\ImagefusionDatasets\StackMFF'
# method_2_compare=["CVT","DWT","DCT","DTCWT","DSIFT","NSCT","IFCNN-MAX","U2Fusion","SDNet","MFF-GAN","StackMFF"]
method_2_compare=["MUFusion","SwinMFF",'DDBFusion']
datasets=['4D-Light-Field','FlyingThings3D','Middlebury','Mobile Depth']
datasets=['Road-MF','NYU Depth V2']
# all_metrics=['NIQE','SF','AVG','EN','STD','MSE','SSIM','PSNR','VIF','MI','CSG']
metrics=['SSIM','PSNR','MSE','MAE','RMSE','logRMS','abs_rel_error','sqr_rel_error','VIF','MI','NIQE','SF','AVG','EN','STD']
metrics=['SSIM','PSNR']
def get_image_formats(folder):
    formats = []
    for root, dirs, files in os.walk(folder):
        for file in files:
            filename, ext = os.path.splitext(file)
            ext = ext[1:].lower() # remove .
            if ext in ['jpg', 'jpeg', 'png', 'gif', 'bmp', 'tiff']:
                if ext not in formats:
                    formats.append(ext)
    return formats[0]
def read_image(path):
    img = cv2.imread(path, 0)
    if img is None:
        # 如果读取失败，尝试更改扩展名
        base, ext = os.path.splitext(path)
        new_ext = '.jpg' if ext.lower() == '.png' else '.png'
        new_path = base + new_ext
        img = cv2.imread(new_path, 0)
    return img
if __name__ == '__main__':
    #获取groundtruth的图片路径
    for dataset in datasets:
        print('testing:',dataset)
        ground_truth_img=natsorted(glob.glob(os.path.join(ground_truth_path,dataset,'AiF','*.jpg')))
        method_results = {}
        df = pd.DataFrame(columns=(['Method']+metrics))
        #遍历方法
        for method_index,method in enumerate(method_2_compare):
            #获取该方法结果的所有图片的路径
            ext='*.'+get_image_formats(os.path.join(base_path,method,dataset))
            img_path_list=natsorted(glob.glob(os.path.join(base_path,method,dataset,ext)))
            metric_results = {}
            #遍历里面的每张图片
            metric_dict={}
            for metric in metrics:
                metric_dict[metric] = 0  # 初始化为0或空值
            for img_truth_ind,img_truth in tqdm(enumerate(ground_truth_img)):

                img_name=os.path.basename(img_truth)
                img_truth = cv2.imread(img_truth, 0)
                result_path = os.path.join(base_path, method, dataset, img_name)
                img_result = read_image(result_path)
                #每张图片都要计算它的不同指标，并累计加上去
                #遍历所有的评测指标
                for metric_index,metric in enumerate(metrics):
                    if metric=='SF':
                        value=SF_function(img_result)

                    if metric=='AVG':
                        value= AG_function(img_result)

                    if metric=='EN':
                        value= EN_function(img_result)

                    if metric=='STD':
                        value= SD_function(img_result)

                    if metric=='MSE':
                        value= MSE_function(img_result,img_truth)

                    if metric=='MAE':
                        value= MAE_function(img_result,img_truth)

                    if metric=='RMSE':
                        value= RMSE_function(img_result,img_truth)

                    if metric=='logRMS':
                        value= logRMS_function(img_result,img_truth)

                    if metric=='abs_rel_error':
                        value= abs_rel_error_function(img_result,img_truth)

                    if metric=='sqr_rel_error':
                        value= sqr_rel_error_function(img_result,img_truth)

                    if metric=='SSIM':
                        value =compare_ssim(img_truth, img_result, multichannel=False)

                    if metric=='PSNR':
                        value =compare_psnr(img_truth, img_result)

                    if metric=="VIF":
                        value = vifp_mscale(img_truth, img_result)

                    if metric == "MI":
                        value = MI_function(img_truth, img_result)

                    if metric =='mean_diff':
                        value = mean_diff(img_truth, img_result)
                    if metric == "NIQE":
                        value = niqe(img_result)

                    if metric == "CSG":

                        def CSG_function(img1,img2):
                            gradient_x_1 = cv2.Sobel(img1, cv2.CV_64F, 1, 0, ksize=3)
                            gradient_y_1 = cv2.Sobel(img1, cv2.CV_64F, 0, 1, ksize=3)
                            gradient_direction_1 = np.arctan2(gradient_y_1, gradient_x_1)
                            gradient_direction_normalized_1 = cv2.normalize(gradient_direction_1, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
                            gradient_x_2 = cv2.Sobel(img2, cv2.CV_64F, 1, 0, ksize=3)
                            gradient_y_2 = cv2.Sobel(img2, cv2.CV_64F, 0, 1, ksize=3)
                            gradient_direction_2 = np.arctan2(gradient_y_2, gradient_x_2)
                            gradient_direction_normalized_2 = cv2.normalize(gradient_direction_2, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
                            result=cosine_similarity(gradient_direction_normalized_1.reshape(1, -1), gradient_direction_normalized_2.reshape(1, -1))[0, 0]
                            return result
                        value = CSG_function(img_result,img_truth)
                    metric_dict[metric] += value

            for metric in metric_dict:
                metric_dict[metric] /= len(img_path_list)
                metric_dict[metric]=round(metric_dict[metric],4)

            method_results[method]=metric_dict

        for key, value in method_results.items():
            print(key, ':', value)
            temp = {'Method': key}
            new_row = {**temp, **value}

            # Create a new DataFrame with the new row
            new_df = pd.DataFrame([new_row])

            # Concatenate the new DataFrame with the existing one
            df = pd.concat([df, new_df], ignore_index=True)

            print(df)
        df.to_excel('compare result_add_{}.xlsx'.format(dataset))

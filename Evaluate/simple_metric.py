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

def AG_function(image):
	width = image.shape[1]
	width = width - 1
	height = image.shape[0]
	height = height - 1
	tmp = 0.0
	[grady, gradx] = np.gradient(image)
	s = np.sqrt((np.square(gradx) + np.square(grady)) / 2)
	AG = np.sum(np.sum(s)) / (width * height)
	return AG

def SF_function(image):
    image_array = np.array(image)
    RF = np.diff(image_array, axis=0)
    RF1 = np.sqrt(np.mean(np.mean(RF ** 2)))
    CF = np.diff(image_array, axis=1)
    CF1 = np.sqrt(np.mean(np.mean(CF ** 2)))
    SF = np.sqrt(RF1 ** 2 + CF1 ** 2)
    return SF

def EN_function(image_array):
    # 计算图像的直方图
    histogram, bins = np.histogram(image_array, bins=256, range=(0, 255))
    # 将直方图归一化
    histogram = histogram / float(np.sum(histogram))
    # 计算熵
    entropy = -np.sum(histogram * np.log2(histogram + 1e-7))
    return entropy

def Mean(image):
    img_array = np.array(image)
    return np.mean(img_array)

def SD_function(image_array):
    m, n = image_array.shape
    u = np.mean(image_array)
    SD = np.sqrt(np.sum(np.sum((image_array - u) ** 2)) / (m * n))
    return SD

def MSE_function(A, F):
    A = A / 255.0
    F = F / 255.0
    m, n = F.shape
    MSE = np.sum(np.sum((F - A)**2))/(m*n)

    return MSE


def RMSE_function(A, F):
    """
    计算两幅图像之间的均方根误差（RMSE）
    :param A: 图像 A 的像素值数组
    :param F: 图像 F 的像素值数组
    :return: 两幅图像之间的 RMSE
    """
    # 将像素值归一化到 [0, 1] 区间
    A = A / 255.0
    F = F / 255.0

    # 获取图像的尺寸
    m, n = F.shape

    # 计算均方误差（MSE）
    MSE = np.sum((F - A) ** 2) / (m * n)

    # 计算均方根误差（RMSE）
    RMSE = np.sqrt(MSE)

    return RMSE


def logRMS_function(A, F):
    """
    计算两幅图像之间的对数均方根误差（logRMS）
    :param A: 图像 A 的像素值数组
    :param F: 图像 F 的像素值数组
    :return: 两幅图像之间的 logRMS
    """
    # 将像素值归一化到 [0, 1] 区间
    A = A / 255.0
    F = F / 255.0

    # 获取图像的尺寸
    m, n = F.shape

    # 计算对数均方根误差（logRMS）
    logRMS = np.sqrt(np.sum((np.log1p(F) - np.log1p(A)) ** 2) / (m * n))

    return logRMS


def abs_rel_error_function(A, F):
    """
    计算两幅图像之间的相对绝对误差（Abs. rel.）
    :param A: 图像 A 的像素值数组
    :param F: 图像 F 的像素值数组
    :return: 两幅图像之间的 Abs. rel.
    """
    # 将像素值归一化到 [0, 1] 区间
    A = A / 255.0
    F = F / 255.0

    # 获取图像的尺寸
    m, n = F.shape

    # 计算相对绝对误差（Abs. rel.）
    abs_rel = np.sum(np.abs(F - A) / (A + 1e-8)) / (m * n)

    return abs_rel


def sqr_rel_error_function(A, F):
    """
    计算两幅图像之间的相对平方误差（Sqr. rel.）
    :param A: 图像 A 的像素值数组
    :param F: 图像 F 的像素值数组
    :return: 两幅图像之间的 Sqr. rel.
    """
    # 将像素值归一化到 [0, 1] 区间
    A = A / 255.0
    F = F / 255.0

    # 获取图像的尺寸
    m, n = F.shape

    # 计算相对平方误差（Sqr. rel.）
    sqr_rel = np.sum(((F - A) ** 2) / (A + 1e-8)) / (m * n)

    return sqr_rel
def MAE_function(A, F):
    # 确保输入是 numpy 数组
    A = np.array(A, dtype=np.float64)
    F = np.array(F, dtype=np.float64)

    # 将像素值归一化到 [0, 1] 范围
    A = A / 255.0
    F = F / 255.0

    # 获取图像尺寸
    m, n = F.shape

    # 计算绝对误差的和
    absolute_diff = np.abs(F - A)

    # 计算平均绝对误差
    MAE = np.sum(absolute_diff) / (m * n)

    return MAE

def mean_diff(A, F):
    # 确保输入是 numpy 数组
    A = np.array(A, dtype=np.float64)
    F = np.array(F, dtype=np.float64)

    # 将像素值归一化到 [0, 1] 范围
    A = A / 255.0
    F = F / 255.0

    # 获取图像尺寸
    m, n = F.shape

    mean_a=np.mean(A)
    mean_F=np.mean(F)
    mean_diff=mean_a/mean_F

    return round(mean_diff,4)
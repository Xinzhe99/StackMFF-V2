#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Xinzhe Xie
# @University  : ZheJiang University

"""
This script generates multi-focus image stacks from single images and their depth maps.
It simulates depth of field effects by applying varying levels of Gaussian blur based on
depth information. The process includes:
1. Depth map quantization
2. Gaussian blur application at different scales
3. Generation of focus stack images
4. Organization of output data structure
"""

import os
import glob
import cv2
import numpy as np
from tqdm import tqdm
import random


def simulate_dof(img, img_depth, num_regions):
    """
    Simulate depth of field effects on an image using its depth map.
    
    Args:
        img (numpy.ndarray): Input image
        img_depth (numpy.ndarray): Corresponding depth map
        num_regions (int): Number of depth regions to simulate
    
    Returns:
        tuple: (list of simulated DOF images, quantized depth map)
    """
    imgs_blurred_list = []
    kernel_list = [2 * i + 1 for i in range(num_regions)]

    for i in kernel_list:
        img_blured = cv2.GaussianBlur(img, (i, i), 0)
        imgs_blurred_list.append(img_blured)

    ref_points = np.linspace(0, 255, num_regions + 1)
    quantized = np.digitize(img_depth, ref_points) - 1
    masks = [(quantized == i) for i in range(num_regions)]

    results = []
    for index_mask, _ in enumerate(masks):
        sys_result = np.zeros_like(img)
        for ind, mask in enumerate(masks):
            target_index = min(abs(ind - index_mask), len(imgs_blurred_list) - 1)
            sys_result[mask] = imgs_blurred_list[target_index][mask]

        # Fill in any remaining black areas with the original image
        black_mask = np.all(sys_result == 0, axis=2)
        sys_result[black_mask] = img[black_mask]

        results.append(sys_result)

    # 创建并返回量化后的深度图
    quantized_depth = (quantized * (255 // (num_regions - 1))).astype(np.uint8)

    return results, quantized_depth


def process_images(original_path, depth_path, output_path):
    """
    Process a set of images and their depth maps to create multi-focus image stacks.
    
    Args:
        original_path (str): Path to original images
        depth_path (str): Path to depth maps
        output_path (str): Path to save generated data
    """
    num_regions_list = [8, 12, 16, 20, 24]
    os.makedirs(output_path, exist_ok=True)

    original_images = glob.glob(os.path.join(original_path, '*.jpg')) + glob.glob(os.path.join(original_path, '*.png'))

    for pic_path in tqdm(original_images, desc="Processing images"):
        filename = os.path.basename(pic_path)
        name, _ = os.path.splitext(filename)

        img = cv2.imread(pic_path)
        img_depth = cv2.imread(os.path.join(depth_path, name + '.png'), 0)

        num_regions = random.choice(num_regions_list)

        dof_results, quantized_depth = simulate_dof(img, img_depth, num_regions)

        # Create separate folders for original, depth, quantized depth, and DOF stack
        original_folder = os.path.join(output_path, 'AiF')
        depth_folder = os.path.join(output_path, 'depth')
        quantized_depth_folder = os.path.join(output_path, 'quantized_depth')
        dof_stack_folder = os.path.join(output_path, 'dof_stack')

        os.makedirs(original_folder, exist_ok=True)
        os.makedirs(depth_folder, exist_ok=True)
        os.makedirs(quantized_depth_folder, exist_ok=True)
        os.makedirs(dof_stack_folder, exist_ok=True)

        # Save the original image
        cv2.imwrite(os.path.join(original_folder, f'{name}.jpg'), img)

        # Save the depth image
        cv2.imwrite(os.path.join(depth_folder, f'{name}.png'), img_depth)

        # Save the quantized depth image
        cv2.imwrite(os.path.join(quantized_depth_folder, f'{name}.png'), quantized_depth)

        # Save the DOF stack
        dof_image_folder = os.path.join(dof_stack_folder, name)
        os.makedirs(dof_image_folder, exist_ok=True)
        for i, result in enumerate(dof_results):
            cv2.imwrite(os.path.join(dof_image_folder, f'{i}.jpg'), result)


if __name__ == "__main__":
    original_path = "./datasets/ADE/test/images"
    depth_path = "./datasets/ADE/test/depth"
    output_path = "./datasets/ADE/test/processed"

    process_images(original_path, depth_path, output_path)
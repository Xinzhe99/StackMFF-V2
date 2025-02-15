#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Xinzhe Xie
# @University  : ZheJiang University

"""
This script processes images from the NYU V2 dataset by cropping them to a specific size.
It handles both RGB images and depth maps, maintaining their aspect ratio and center alignment.
The cropping is done to remove boundary artifacts and standardize image dimensions.
"""

import os
import cv2
import numpy as np
from tqdm import tqdm

def process_images(input_folder, output_folder, crop_size):
    """
    Process and crop images from the input folder to the specified size.
    
    Args:
        input_folder (str): Path to the folder containing original images
        output_folder (str): Path to save the cropped images
        crop_size (tuple): Target size for cropping (height, width)
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Get list of image files
    image_files = [f for f in os.listdir(input_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]

    for image_file in tqdm(image_files, desc="Processing images"):
        # Read the image
        image_path = os.path.join(input_folder, image_file)
        img = cv2.imread(image_path)

        # Calculate crop dimensions
        h, w = img.shape[:2]
        crop_h, crop_w = crop_size
        start_h = (h - crop_h) // 2
        start_w = (w - crop_w) // 2
        
        # Crop the image from center
        cropped_img = img[start_h:start_h + crop_h, start_w:start_w + crop_w]

        # Save the cropped image
        output_path = os.path.join(output_folder, image_file)
        cv2.imwrite(output_path, cropped_img)

if __name__ == "__main__":
    # Set crop size based on depth map dimensions
    crop_size = (460, 620)

    # Process RGB images
    input_folder = "./datasets/NYU_V2/rgb_images"
    output_folder = "./datasets/NYU_V2/rgb_images_cropped"
    process_images(input_folder, output_folder, crop_size)

    # Process depth maps
    input_folder = "./datasets/NYU_V2/depth_maps"
    output_folder = "./datasets/NYU_V2/depth_maps_cropped"
    process_images(input_folder, output_folder, crop_size)

    print("Image processing completed!")
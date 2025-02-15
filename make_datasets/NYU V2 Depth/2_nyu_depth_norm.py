#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Xinzhe Xie
# @University  : ZheJiang University

"""
This script normalizes depth maps from the NYU V2 dataset.
It converts the raw depth values to a normalized range of 0-255,
making them suitable for visualization and processing.
The normalization preserves relative depth relationships while
ensuring consistent value ranges across all images.
"""

import os
import cv2
import numpy as np
from tqdm import tqdm

def normalize_depth_maps(input_folder, output_folder):
    """
    Normalize depth maps to the range [0, 255].
    
    Args:
        input_folder (str): Directory containing original depth maps
        output_folder (str): Directory to save normalized depth maps
    """
    # Ensure output directory exists
    os.makedirs(output_folder, exist_ok=True)

    # Process all images in the input folder
    for filename in tqdm(os.listdir(input_folder), desc="Normalizing depth maps"):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            # Read depth map with original values
            img_path = os.path.join(input_folder, filename)
            depth_img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

            # Convert to float32 for processing
            depth_img = depth_img.astype(np.float32)

            # Normalize depth values to [0, 255] range
            normalized_depth = cv2.normalize(depth_img, None, 0, 255, cv2.NORM_MINMAX)

            # Convert to uint8 for saving
            normalized_depth = normalized_depth.astype(np.uint8)

            # Save normalized depth map
            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, normalized_depth)

if __name__ == "__main__":
    # Set input and output directories
    input_folder = "./datasets/NYU_V2/depth_maps_cropped"
    output_folder = "./datasets/NYU_V2/depth_maps_normalized"

    # Process depth maps
    normalize_depth_maps(input_folder, output_folder)
    print("Depth map normalization completed successfully.")
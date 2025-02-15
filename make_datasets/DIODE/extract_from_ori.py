#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Xinzhe Xie
# @University  : ZheJiang University

"""
This script extracts and processes images from the DIODE dataset.
It converts PNG images that have corresponding depth maps to JPG format.
The script maintains the original image quality while reducing storage space.
"""

import os
import cv2
from tqdm import tqdm

def process_dataset(source_dir, image_dest_dir):
    """
    Process images from the DIODE dataset and convert them to JPG format.
    
    Args:
        source_dir (str): Source directory containing the original dataset
        image_dest_dir (str): Destination directory for processed images
    """
    for root, dirs, files in tqdm(os.walk(source_dir)):
        for file in files:
            if file.endswith('_depth.npy'):
                # Find corresponding image file
                image_file = file.replace('_depth.npy', '.png')
                src_path = os.path.join(root, image_file)
                
                # Process image if it exists
                if os.path.exists(src_path):
                    img = cv2.imread(src_path)
                    if img is not None:
                        # Convert and save as JPG
                        dst_path = os.path.join(image_dest_dir, image_file.replace('.png', '.jpg'))
                        cv2.imwrite(dst_path, img)

if __name__ == "__main__":
    # Define source and destination directories
    source_dir = "./datasets/DIODE/train"
    image_dest_dir = "./datasets/DIODE/train_images"

    # Create destination directory if it doesn't exist
    os.makedirs(image_dest_dir, exist_ok=True)

    # Process the dataset
    process_dataset(source_dir, image_dest_dir)
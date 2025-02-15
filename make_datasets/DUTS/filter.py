#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Xinzhe Xie
# @University  : ZheJiang University

"""
This script filters images based on background uniformity.
It separates images into two categories:
1. Images with uniform backgrounds (rejected)
2. Images with diverse backgrounds (filtered)

The filtering is based on the ratio of the most common color in the image.
"""

from PIL import Image
import os
from collections import Counter
from tqdm import tqdm
import shutil

def is_mostly_uniform_background(image_path, threshold):
    """
    Determine if an image has a mostly uniform background.

    Args:
        image_path (str): Path to the image file
        threshold (float): Threshold for determining background uniformity (0-1)

    Returns:
        bool: True if the image has a mostly uniform background, False otherwise
    """
    # Open and convert image to RGB
    image = Image.open(image_path)
    image = image.convert('RGB')
    
    # Get pixel data
    pixels = image.getdata()

    # Count color frequencies
    color_counter = Counter(pixels)

    # Get the most common color and its count
    most_common_color, most_common_count = color_counter.most_common(1)[0]

    # Calculate the ratio of the most common color
    most_common_ratio = most_common_count / len(pixels)

    # Return True if the ratio exceeds the threshold
    return most_common_ratio >= threshold

def filter_uniform_background_images(input_dir, output_dir_filtered, output_dir_rejected, threshold):
    """
    Filter images based on background uniformity and save them to separate directories.

    Args:
        input_dir (str): Directory containing input images
        output_dir_filtered (str): Directory for saving images with diverse backgrounds
        output_dir_rejected (str): Directory for saving images with uniform backgrounds
        threshold (float): Threshold for determining background uniformity (0-1)
    """
    # Ensure output directories exist
    if not os.path.exists(output_dir_filtered):
        os.makedirs(output_dir_filtered)
    if not os.path.exists(output_dir_rejected):
        os.makedirs(output_dir_rejected)

    # Get all image files from input directory
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]

    # Initialize counters
    total_images = len(image_files)
    rejected_images = 0

    # Process each image with progress bar
    for filename in tqdm(image_files, desc="Processing images"):
        image_path = os.path.join(input_dir, filename)
        if is_mostly_uniform_background(image_path, threshold):
            # Save to rejected directory if background is uniform
            output_path = os.path.join(output_dir_rejected, filename)
            rejected_images += 1
        else:
            # Save to filtered directory if background is diverse
            output_path = os.path.join(output_dir_filtered, filename)

        # Copy image to appropriate output directory
        shutil.copy2(image_path, output_path)

    # Print filtering statistics
    filter_ratio = rejected_images / total_images
    print(f"Filtered out {rejected_images} out of {total_images} images ({filter_ratio * 100:.2f}%).")

# Example usage
if __name__ == "__main__":
    # Set filtering threshold
    threshold = 0.2

    # Process training set
    input_directory = './datasets/DUTS/DUTS-TR/images'
    output_directory_filtered = './datasets/DUTS/DUTS-TR/images_filtered'
    output_directory_rejected = './datasets/DUTS/DUTS-TR/images_rejected'
    filter_uniform_background_images(input_directory, output_directory_filtered, output_directory_rejected, threshold=threshold)

    # Process test set
    input_directory = './datasets/DUTS/DUTS-TE/images'
    output_directory_filtered = './datasets/DUTS/DUTS-TE/images_filtered'
    output_directory_rejected = './datasets/DUTS/DUTS-TE/images_rejected'
    filter_uniform_background_images(input_directory, output_directory_filtered, output_directory_rejected, threshold=threshold)

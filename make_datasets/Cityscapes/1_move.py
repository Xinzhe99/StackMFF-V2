#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Xinzhe Xie
# @University  : ZheJiang University

"""
This script reorganizes the Cityscapes dataset by moving images from their
original hierarchical structure to a flattened directory structure.
It processes train, validation, and test sets separately while maintaining
their split organization.
"""

import os
import shutil
from tqdm import tqdm

def reorganize_cityscapes(source_base_path, target_base_path):
    """
    Reorganize Cityscapes dataset by flattening the directory structure.
    
    Args:
        source_base_path (str): Path to original Cityscapes dataset
        target_base_path (str): Path to save reorganized dataset
    """
    # Define dataset splits
    subfolders = ["train", "test", "val"]

    # Create target directories
    for subfolder in subfolders:
        target_folder = os.path.join(target_base_path, subfolder)
        os.makedirs(target_folder, exist_ok=True)

    # Process each split
    for subfolder in subfolders:
        source_folder = os.path.join(source_base_path, subfolder)
        target_folder = os.path.join(target_base_path, subfolder)

        # Get total file count for progress bar
        total_files = sum([len(files) for _, _, files in os.walk(source_folder)])
        
        # Initialize progress bar
        with tqdm(total=total_files, desc=f"Processing {subfolder} set") as pbar:
            # Walk through directory structure
            for root, _, files in os.walk(source_folder):
                for file in files:
                    # Get source and destination paths
                    source_path = os.path.join(root, file)
                    
                    # Copy file to flattened structure
                    shutil.copy(source_path, target_folder)
                    pbar.update(1)

if __name__ == "__main__":
    # Define source and target paths
    source_base_path = "./datasets/Cityscapes/leftImg8bit"
    target_base_path = "./datasets/Cityscapes/processed"

    # Execute reorganization
    reorganize_cityscapes(source_base_path, target_base_path)
    print("Dataset reorganization completed successfully.")
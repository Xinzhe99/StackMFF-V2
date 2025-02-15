#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Xinzhe Xie
# @University  : ZheJiang University

"""
This script splits the NYU V2 dataset into training and testing sets.
It handles three types of data:
1. Multi-focus image stacks (dof_stack)
2. All-in-Focus images (AiF)
3. Depth maps (depth)

The split maintains the relationship between corresponding files
across all three data types.
"""

import os
import random
import shutil

def split_dataset(src_dir, train_dir, test_dir, test_size=100):
    """
    Split the dataset into training and testing sets.
    
    Args:
        src_dir (str): Source directory containing the complete dataset
        train_dir (str): Directory to save training data
        test_dir (str): Directory to save testing data
        test_size (int): Number of samples for the test set (default: 100)
    """
    # Get all scene folders from dof_stack directory
    dof_stack_folders = [f for f in os.listdir(os.path.join(src_dir, 'dof_stack')) if
                        os.path.isdir(os.path.join(src_dir, 'dof_stack', f))]

    # Randomly select test set folders
    test_folders = random.sample(dof_stack_folders, test_size)

    # Create directory structure for both sets
    for dir_name in [train_dir, test_dir]:
        for subdir in ['AiF', 'depth', 'dof_stack']:
            os.makedirs(os.path.join(dir_name, subdir), exist_ok=True)

    # Copy files to respective directories
    for folder in dof_stack_folders:
        # Determine destination directory
        dest_dir = test_dir if folder in test_folders else train_dir

        # Copy multi-focus stack
        src_dof_stack = os.path.join(src_dir, 'dof_stack', folder)
        dest_dof_stack = os.path.join(dest_dir, 'dof_stack', folder)
        shutil.copytree(src_dof_stack, dest_dof_stack)

        # Copy corresponding AiF and depth files
        src_aif = os.path.join(src_dir, 'AiF', f"{folder}.jpg")
        dest_aif = os.path.join(dest_dir, 'AiF', f"{folder}.jpg")
        src_depth = os.path.join(src_dir, 'depth', f"{folder}.png")
        dest_depth = os.path.join(dest_dir, 'depth', f"{folder}.png")

        # Copy AiF file if exists
        if os.path.exists(src_aif):
            shutil.copy2(src_aif, dest_aif)
        else:
            print(f"Warning: AiF file {src_aif} not found.")

        # Copy depth file if exists
        if os.path.exists(src_depth):
            shutil.copy2(src_depth, dest_depth)
        else:
            print(f"Warning: Depth file {src_depth} not found.")

    print(f"Dataset split complete:")
    print(f"- Test set: {len(test_folders)} scenes")
    print(f"- Training set: {len(dof_stack_folders) - len(test_folders)} scenes")

if __name__ == "__main__":
    # Set source and destination directories
    src_dir = "./datasets/NYU_V2/processed"
    train_dir = "./datasets/NYU_V2/train"
    test_dir = "./datasets/NYU_V2/test"

    # Execute dataset splitting
    split_dataset(src_dir, train_dir, test_dir)
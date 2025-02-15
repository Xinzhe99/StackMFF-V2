#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Xinzhe Xie
# @University  : ZheJiang University

"""
This script extracts and copies all JPG files from a source directory to a destination directory.
It maintains the original file quality and metadata during the copying process.
"""

import os
import shutil

def copy_jpg_files(source_folder, destination_folder):
    """
    Copy all JPG files from source folder to destination folder.
    
    Args:
        source_folder (str): Path to the source directory containing JPG files
        destination_folder (str): Path to the destination directory where files will be copied
    """
    # Ensure destination folder exists, create if not
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # Walk through all files and subdirectories in source folder
    for root, dirs, files in os.walk(source_folder):
        for file in files:
            # Check if file is a JPG
            if file.lower().endswith('.jpg'):
                # Construct full path for source file
                source_file = os.path.join(root, file)
                # Construct full path for destination file
                destination_file = os.path.join(destination_folder, file)
                # Copy file while preserving metadata
                shutil.copy2(source_file, destination_file)
                print(f"Copied {source_file} to {destination_file}")

if __name__ == "__main__":
    # Specify source and destination folders
    source_folder = "./datasets/ADE20K/images/validation"
    destination_folder = "./datasets/ADE/ADE-TE"

    # Execute the file copying process
    copy_jpg_files(source_folder, destination_folder)
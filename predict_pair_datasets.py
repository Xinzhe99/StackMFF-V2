# -*- coding: utf-8 -*-
# @Author  : XinZhe Xie
# @University  : ZheJiang University

import argparse
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
from tqdm import tqdm
from network import StackMFF_V2
import random
import re
import time
import cv2
import numpy as np
import torch.nn.functional as F
import os
import matplotlib
import torch
import torch.nn as nn

def parse_args():
    """
    Parse command line arguments for batch evaluation of multiple pair datasets
    
    Returns:
        argparse.Namespace: Parsed arguments containing:
            - model_path: Path to the trained model weights
            - test_root: Root directory containing test datasets
            - test_datasets: List of dataset names to evaluate
            - batch_size: Batch size for evaluation
            - num_workers: Number of data loading workers
            - output_dir: Directory for saving results
    """
    parser = argparse.ArgumentParser(description="Batch evaluation script for multiple pair datasets")
    parser.add_argument('--model_path', type=str, default='model.pth',
                        help='Path to the trained model weights')
    parser.add_argument('--test_root', type=str, required=True,
                        help='Path to test data root directory')
    parser.add_argument('--test_datasets', nargs='+',
                        default=['dataset1', 'dataset2'],
                        help='List of test datasets to evaluate')
    parser.add_argument('--batch_size', type=int, default=1, 
                        help='Batch size for evaluation')
    parser.add_argument('--num_workers', type=int, default=0, 
                        help='Number of data loading workers')
    parser.add_argument('--output_dir', type=str, default='./output_pair',
                        help='Directory for saving evaluation results')
    parser.add_argument('--reverse', action='store_true', default=False,
                        help='Reverse depth maps and index maps (depth = 1 - depth, index = max_index - index)')
    return parser.parse_args()

def config_model_dir(resume=False,subdir_name='train_runs'):
    # 获取当前项目目录
    project_dir = os.getcwd()
    # 获取models文件夹的路径
    models_dir = os.path.join(project_dir, subdir_name)
    # 如果models文件夹不存在，就创建它
    if not os.path.exists(models_dir):
        os.mkdir(models_dir)
    # 如果是第一次运行 创建models0
    if not os.path.exists(os.path.join(models_dir,subdir_name+'1')):
        os.mkdir(os.path.join(models_dir,subdir_name+'1'))
        return os.path.join(models_dir,subdir_name+'1')
    else:
        # 获取models文件夹下已有的子文件夹
        sub_dirs = [d for d in os.listdir(models_dir) if os.path.isdir(os.path.join(models_dir, d))]
        sub_dirs.sort(key=lambda l: int(re.findall('\d+', l)[0]))
        last_numbers=re.findall("\d+",sub_dirs[-1])#list
        if resume==False:
            new_sub_dir_name = subdir_name + str(int(last_numbers[0]) + 1)
        else:
            new_sub_dir_name = subdir_name + str(int(last_numbers[0]))
        model_dir_path = os.path.join(models_dir, new_sub_dir_name)
        if resume == False:
            os.mkdir(model_dir_path)
        else:
            pass
        # 把创建的路径存在model_save变量里
        print(model_dir_path)
        return model_dir_path
        
def resize_to_multiple_of_32(image):
    """
    Resize input image tensor to dimensions that are multiples of 32
    
    Args:
        image (torch.Tensor): Input image tensor
        
    Returns:
        tuple: (resized_image, (original_height, original_width))
            - resized_image: Image resized to multiple of 32
            - tuple of original dimensions
    """
    h, w = image.shape[-2:]
    new_h = ((h - 1) // 32 + 1) * 32
    new_w = ((w - 1) // 32 + 1) * 32
    resized_image = F.interpolate(image, size=(new_h, new_w), mode='bilinear', align_corners=False)
    return resized_image, (h, w)

def gray_to_colormap(img, cmap='rainbow'):
    """
    Convert grayscale image to colormap visualization
    
    Args:
        img (numpy.ndarray): Input grayscale image (normalized to [0,1])
        cmap (str): Matplotlib colormap name
        
    Returns:
        numpy.ndarray: RGB colormap visualization
    """
    img = np.clip(img, 0, 1)
    norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
    cmap_m = matplotlib.cm.get_cmap(cmap)
    map = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap_m)
    colormap = (map.to_rgba(img)[:, :, :3] * 255).astype(np.uint8)
    return colormap

class TestPairDataset(Dataset):
    """
    Dataset class for loading and processing image pairs from a single dataset
    
    Args:
        root_dir (str): Root directory containing A and B folders for a single dataset
        transform (callable, optional): Optional transform to be applied on images
        subset_fraction (float, optional): Fraction of total pairs to use
    """
    def __init__(self, root_dir, transform=None, subset_fraction=1.0):
        self.root_dir = root_dir
        self.transform = transform
        self.image_pairs = []
        self.pair_names = []
        
        # 支持的图像格式扩展列表
        self.supported_formats = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.webp', '.ppm', '.pgm', '.pbm')

        # Check if A and B folders exist in the dataset directory
        folder_a = os.path.join(root_dir, 'A')
        folder_b = os.path.join(root_dir, 'B')
        
        if os.path.exists(folder_a) and os.path.exists(folder_b):
            # Get all images in folder A and B with extended format support
            images_a = sorted([f for f in os.listdir(folder_a) 
                             if f.lower().endswith(self.supported_formats)], 
                             key=self.sort_key)
            images_b = sorted([f for f in os.listdir(folder_b) 
                             if f.lower().endswith(self.supported_formats)], 
                             key=self.sort_key)
            
            # Create pairs based on matching indices
            min_length = min(len(images_a), len(images_b))
            for i in range(min_length):
                img_a_path = os.path.join(folder_a, images_a[i])
                img_b_path = os.path.join(folder_b, images_b[i])
                
                # Create a pair as a "stack" of 2 images
                pair = [img_a_path, img_b_path]
                pair_name = f"{i+1}"  # Simple numbering for pairs within this dataset
                
                self.image_pairs.append(pair)
                self.pair_names.append(pair_name)
            
            print(f"Found {len(images_a)} images in A, {len(images_b)} images in B, created {min_length} pairs")
        else:
            print(f'Warning: A or B folder not found in {root_dir}')

        # Apply subset sampling if requested
        if subset_fraction < 1.0:
            subset_size = int(len(self.image_pairs) * subset_fraction)
            indices = random.sample(range(len(self.image_pairs)), subset_size)
            self.image_pairs = [self.image_pairs[i] for i in indices]
            self.pair_names = [self.pair_names[i] for i in indices]

        print(f"Total loaded {len(self.image_pairs)} image pairs")
        print(f"Supported image formats: {', '.join(self.supported_formats)}")

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        """
        Get a single image pair as a "stack"
        
        Args:
            idx (int): Index of the pair
            
        Returns:
            tuple: (pair_tensor, pair_name, num_images)
                - pair_tensor: Tensor containing the image pair (2 images)
                - pair_name: Name of the pair
                - num_images: Number of images in the pair (always 2)
        """
        image_pair = self.image_pairs[idx]
        pair_name = self.pair_names[idx]

        images = []
        for img_path in image_pair:
            # Read image and convert to grayscale
            bgr_img = cv2.imread(img_path)
            if bgr_img is None:
                print(f"Warning: Could not read image {img_path}")
                # Create a dummy image if reading fails
                bgr_img = np.zeros((256, 256, 3), dtype=np.uint8)
            
            gray_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
            # Normalize to [0,1]
            gray_img = gray_img.astype(np.float32) / 255.0
            if self.transform:
                gray_img = self.transform(gray_img)
            images.append(gray_img.squeeze(0))

        pair_tensor = torch.stack(images)
        return pair_tensor, pair_name, len(images)

    @staticmethod
    def sort_key(filename):
        """
        Extract numerical value from filename for sorting
        """
        numbers = re.findall(r'\d+\.?\d*', filename)
        return float(numbers[0]) if numbers else 0


def infer_dataset(model, dataset_loader, device, save_path, reverse=False):
    """
    Perform inference on a dataset and save results
    
    Args:
        model: Neural network model
        dataset_loader: DataLoader containing the test dataset
        device: Computing device (CPU/GPU)
        save_path: Directory to save results for this specific dataset
        reverse: Whether to reverse depth maps and index maps
    
    Returns:
        float: Average inference time per pair
    """
    model.eval()
    
    # Create output subdirectories for this dataset
    subdirs = ['fused_images', 'depth_maps', 'color_fused_images', 'depth_colormaps', 'depth_indices', 'depth_indices_vis']
    for subdir in subdirs:
        os.makedirs(os.path.join(save_path, subdir), exist_ok=True)

    # Initialize timing metrics
    total_inference_time = 0
    total_pairs = 0

    with torch.no_grad():
        for idx, (image_pair, pair_name, _) in tqdm(enumerate(dataset_loader)):
            # Save original dimensions
            original_size = image_pair.shape[-2:]

            # Resize input to multiple of 32 for network processing
            resized_image_pair, _ = resize_to_multiple_of_32(image_pair)
            resized_image_pair = resized_image_pair.to(device)

            # Load original color images for color fusion
            color_pair = load_color_stack(dataset_loader.dataset.image_pairs[idx])

            # Measure inference time
            start_time = time.time()
            fused_image, estimated_depth, depth_map_index = model(resized_image_pair)
            end_time = time.time()

            inference_time = end_time - start_time
            total_inference_time += inference_time
            total_pairs += 1

            # Process model outputs
            fused_image, estimated_depth, depth_map_index = process_model_output(
                fused_image, estimated_depth, depth_map_index, original_size)

            # Create color fused image using original depth_map_index (before any reverse transformation)
            color_fused_bgr = create_color_fused_image(fused_image, depth_map_index, color_pair)
            
            # Prepare depth and index for saving (apply reverse if requested)
            save_depth = estimated_depth
            save_index = depth_map_index
            if reverse:
                save_depth = 1.0 - estimated_depth
                max_index = len(color_pair) - 1
                save_index = max_index - depth_map_index
            
            # Create depth colormap using the depth for saving
            depth_colormap = gray_to_colormap(save_depth)
            depth_colormap_bgr = cv2.cvtColor(depth_colormap, cv2.COLOR_RGB2BGR)
            
            # Generate output filename without dataset prefix since each dataset has its own folder
            filename = f'{pair_name[0]}.png'
            filename_npy = f'{pair_name[0]}.npy'
            
            # Save all results
            try:
                cv2.imwrite(os.path.join(save_path, subdirs[0], filename), 
                           (fused_image * 255).astype(np.uint8))
                cv2.imwrite(os.path.join(save_path, subdirs[1], filename), 
                           (save_depth * 255).astype(np.uint8))
                cv2.imwrite(os.path.join(save_path, subdirs[2], filename), 
                           color_fused_bgr)
                cv2.imwrite(os.path.join(save_path, subdirs[3], filename), 
                           depth_colormap_bgr)
                
                # Save index map as npy format (using save_index)
                np.save(os.path.join(save_path, subdirs[4], filename_npy), save_index)
                
                # Save normalized visualization of index map (using save_index)
                normalized_index = (save_index / (len(color_pair) - 1) * 255).astype(np.uint8)
                cv2.imwrite(os.path.join(save_path, subdirs[5], filename), normalized_index)
            except Exception as e:
                print(f"Error saving images: {str(e)}")
                continue

    # Calculate average inference time
    avg_inference_time = total_inference_time / total_pairs if total_pairs > 0 else 0
    return avg_inference_time


def main():
    """
    Main function to run batch evaluation
    
    Directory structure should be:
    test_root/
        dataset1/
            A/
                1.png (or .jpg, .jpeg, .bmp, .tiff, .tif, .webp, .ppm, .pgm, .pbm)
                2.png
                ...
            B/
                1.png (or .jpg, .jpeg, .bmp, .tiff, .tif, .webp, .ppm, .pgm, .pbm)
                2.png
                ...
        dataset2/
            A/
                1.jpg
                2.jpg
                ...
            B/
                1.jpg
                2.jpg
                ...
    """
    # Parse command line arguments
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Set up device and model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = StackMFF_V2()
    # Load model with proper device mapping for CPU/GPU compatibility
    if torch.cuda.is_available():
        state_dict = torch.load(args.model_path)
    else:
        state_dict = torch.load(args.model_path, map_location=torch.device('cpu'))
    
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)
    model.to(device)
    
    # Only use DataParallel if CUDA is available and multiple GPUs exist
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    elif torch.cuda.is_available():
        # Single GPU, no need for DataParallel
        pass
    # For CPU, no DataParallel needed

    # Initialize test data loaders for each dataset
    test_loaders = {}
    for dataset_name in args.test_datasets:
        dataset_root = os.path.join(args.test_root, dataset_name)
        if not os.path.exists(dataset_root):
            print(f"Warning: Dataset directory {dataset_root} not found. Skipping...")
            continue

        # Set up data transforms
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        # Create dataset and dataloader for this specific dataset
        dataset = TestPairDataset(
            root_dir=dataset_root,
            transform=transform
        )

        test_loaders[dataset_name] = DataLoader(
            dataset, 
            shuffle=False, 
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )

    # 创建带时间戳的统一结果目录
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    unified_results_dir = os.path.join(args.output_dir, f'results_{timestamp}')
    os.makedirs(unified_results_dir, exist_ok=True)

    # Process each dataset
    dataset_avg_times = {}
    for dataset, loader in test_loaders.items():
        # 每个数据集在统一结果目录下创建子文件夹
        dataset_save_path = os.path.join(unified_results_dir, dataset)
        os.makedirs(dataset_save_path, exist_ok=True)
        print(f"Processing dataset: {dataset}")
        avg_time = infer_dataset(model, loader, device, dataset_save_path, args.reverse)
        dataset_avg_times[dataset] = avg_time

    # Print results summary
    print(f"\nResults saved to: {unified_results_dir}")
    print("\nAverage inference times per image pair:")
    for dataset, avg_time in dataset_avg_times.items():
        print(f"{dataset}: {avg_time:.4f} seconds")


def create_color_fused_image(fused_image, depth_map_index, color_pair):
    """
    Create color fused image based on depth map index (vectorized implementation)
    
    Args:
        fused_image: Fused grayscale image
        depth_map_index: Index map indicating which image to sample from
        color_pair: List of original color images (2 images)
    
    Returns:
        numpy.ndarray: Color fused image in BGR format
    """
    height, width = fused_image.shape
    num_images = len(color_pair)

    # Ensure indices are within valid range
    depth_map_index = np.clip(depth_map_index, 0, num_images - 1).astype(int)

    # Convert color pair to single numpy array (N, H, W, 3)
    color_array = np.stack(color_pair, axis=0)

    # Use advanced indexing to get corresponding color values
    fused_color = color_array[depth_map_index, np.arange(height)[:, None], np.arange(width)]

    # Convert RGB to BGR for saving
    fused_color_bgr = cv2.cvtColor(fused_color.astype(np.uint8), cv2.COLOR_RGB2BGR)
    return fused_color_bgr


def load_color_stack(image_paths):
    """
    Load color image pair from file paths
    
    Args:
        image_paths: List of paths to image files
    
    Returns:
        list: List of RGB images
    """
    color_images = []
    for img_path in image_paths:
        # Read BGR image and convert to RGB
        bgr_img = cv2.imread(img_path)
        if bgr_img is None:
            print(f"Warning: Could not read color image {img_path}")
            # Create a dummy image if reading fails
            bgr_img = np.zeros((256, 256, 3), dtype=np.uint8)
        rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        color_images.append(rgb_img)
    return color_images


def process_model_output(fused_image, estimated_depth, depth_map_index, original_size):
    """
    Process model outputs and resize to original dimensions
    
    Args:
        fused_image: Fused image from model
        estimated_depth: Estimated depth map from model
        depth_map_index: Depth index map from model
        original_size: Original image dimensions
    
    Returns:
        tuple: (fused_image, estimated_depth, depth_map_index) resized to original dimensions
    """
    # Convert to numpy and resize back to original size
    fused_image = cv2.resize(fused_image.cpu().numpy().squeeze(),
                             (original_size[1], original_size[0]))
    estimated_depth = cv2.resize(estimated_depth.cpu().numpy().squeeze(),
                                 (original_size[1], original_size[0]))
    depth_map_index = cv2.resize(depth_map_index.cpu().numpy().squeeze(),
                                 (original_size[1], original_size[0]),
                                 interpolation=cv2.INTER_NEAREST)

    return fused_image, estimated_depth, depth_map_index


if __name__ == "__main__":
    main()

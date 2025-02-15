# -*- coding: utf-8 -*-
# @Author  : XinZhe Xie
# @University  : ZheJiang University

import os
import random
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from torchvision import transforms
from PIL import Image
from collections import defaultdict
import torchvision.transforms.functional as TF
from torch.utils.data import ConcatDataset

class ImageStackDataset(Dataset):
    """
    Dataset class for handling stacks of images and their corresponding depth maps.
    Supports data augmentation and subset sampling.
    """
    def __init__(self, root_dir, continuous_depth_dir, transform=None, augment=True, subset_fraction=1):
        """
        Initialize the dataset.
        Args:
            root_dir: Directory containing image stacks
            continuous_depth_dir: Directory containing depth maps
            transform: Optional transforms to be applied
            augment: Whether to apply data augmentation
            subset_fraction: Fraction of the dataset to use (0-1)
        """
        self.root_dir = root_dir
        self.continuous_depth_dir = continuous_depth_dir
        self.transform = transform
        self.augment = augment
        self.image_stacks = []
        self.continuous_depth_maps = []
        self.stack_sizes = []

        all_stacks = sorted(os.listdir(root_dir))
        subset_size = int(len(all_stacks) * subset_fraction)
        selected_stacks = random.sample(all_stacks, subset_size)

        for stack_name in selected_stacks:
            stack_path = os.path.join(root_dir, stack_name)
            if os.path.isdir(stack_path):
                image_stack = []
                for img_name in sorted(os.listdir(stack_path), key=self.sort_key):
                    if img_name.lower().endswith(('.png', '.jpg', '.bmp')):
                        img_path = os.path.join(stack_path, img_name)
                        image_stack.append(img_path)

                if image_stack:
                    continuous_depth_map_path = os.path.join(continuous_depth_dir, stack_name + '.png')
                    if os.path.exists(continuous_depth_map_path):
                        self.image_stacks.append(image_stack)
                        self.continuous_depth_maps.append(continuous_depth_map_path)
                        self.stack_sizes.append(len(image_stack))
                    else:
                        print(f"Warning: Depth map not found for {stack_name}")
                else:
                    print(f'Failed to read image stack: {stack_name}')

    def __len__(self):
        return len(self.image_stacks)

    def __getitem__(self, idx):
        """
        Get a single item from the dataset.
        Returns:
            stack_tensor: Tensor of stacked images (N, H, W)
            continuous_depth_map: Corresponding depth map
            len(images): Number of images in the stack
        """
        image_stack = self.image_stacks[idx]
        continuous_depth_map_path = self.continuous_depth_maps[idx]

        images = []
        for img_path in image_stack:
            image = Image.open(img_path).convert('YCbCr')
            image = image.split()[0]  # 只保留 Y 通道
            images.append(image)

        continuous_depth_map = Image.open(continuous_depth_map_path).convert('L')

        if self.augment:
            images, continuous_depth_map = self.consistent_transform(images, continuous_depth_map)

        # 应用其他变换
        if self.transform:
            images = [self.transform(img) for img in images]
            continuous_depth_map = self.transform(continuous_depth_map)

        # 转换为张量并移除通道维度
        images = [img.squeeze(0) for img in images]
        stack_tensor = torch.stack(images)  # 形状将是 (N, H, W)

        return stack_tensor, continuous_depth_map, len(images)

    def consistent_transform(self, images, continuous_depth_map):
        """
        Apply consistent transformations to both images and depth map.
        Includes random horizontal and vertical flips.
        """
        # 随机水平翻转
        if random.random() > 0.5:
            images = [TF.hflip(img) for img in images]
            continuous_depth_map = TF.hflip(continuous_depth_map)

        # 随机垂直翻转
        if random.random() > 0.5:
            images = [TF.vflip(img) for img in images]
            continuous_depth_map = TF.vflip(continuous_depth_map)

        return images, continuous_depth_map

    @staticmethod
    def sort_key(filename):
        """
        Helper function to sort filenames based on their numerical values.
        """
        return int(''.join(filter(str.isdigit, filename)))

class GroupedBatchSampler(Sampler):
    """
    Custom batch sampler that groups samples by stack size for efficient batching.
    Ensures that each batch contains stacks of the same size.
    """
    def __init__(self, stack_sizes, batch_size):
        """
        Initialize the sampler.
        Args:
            stack_sizes: List of stack sizes for each sample
            batch_size: Number of samples per batch
        """
        self.stack_size_groups = defaultdict(list)
        for idx, size in enumerate(stack_sizes):
            self.stack_size_groups[size].append(idx)
        self.batch_size = batch_size
        self.batches = self._create_batches()

    def _create_batches(self):
        """
        Create batches of indices grouped by stack size.
        Returns shuffled batches for random sampling.
        """
        batches = []
        for size, indices in self.stack_size_groups.items():
            for i in range(0, len(indices), self.batch_size):
                batches.append(indices[i:i + self.batch_size])
        random.shuffle(batches)
        return batches

    def __iter__(self):
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)

def get_updated_dataloader(dataset_params, batch_size, num_workers=4, augment=True, target_size=384):
    """
    Create a DataLoader with multiple datasets combined.
    Args:
        dataset_params: List of parameter dictionaries for each dataset
        batch_size: Number of samples per batch
        num_workers: Number of worker processes for data loading
        augment: Whether to apply data augmentation
        target_size: Size to resize images to
    Returns:
        DataLoader object with combined datasets
    """
    transform = transforms.Compose([
        transforms.Resize((target_size, target_size)),
        transforms.ToTensor(),
    ])

    datasets = []
    for params in dataset_params:
        dataset = ImageStackDataset(
            root_dir=params['root_dir'],
            continuous_depth_dir=params['continuous_depth_dir'],
            transform=transform,
            augment=augment,
            subset_fraction=params['subset_fraction']
        )
        datasets.append(dataset)

    combined_dataset = CombinedDataset(datasets)

    sampler = GroupedBatchSampler(combined_dataset.stack_sizes, batch_size)

    dataloader = DataLoader(combined_dataset, batch_sampler=sampler, num_workers=num_workers)
    return dataloader

class CombinedDataset(ConcatDataset):
    """
    Extension of ConcatDataset that maintains stack size information
    when combining multiple datasets.
    """
    def __init__(self, datasets):
        """
        Initialize the combined dataset.
        Args:
            datasets: List of UpdatedImageStackDataset objects to combine
        """
        super(CombinedDataset, self).__init__(datasets)
        self.stack_sizes = []
        for dataset in datasets:
            self.stack_sizes.extend(dataset.stack_sizes)

    def __getitem__(self, idx):
        return super(CombinedDataset, self).__getitem__(idx)
# -*- coding: utf-8 -*-
# @Author  : XinZhe Xie
# @University  : ZheJiang University

from torchvision.utils import save_image, make_grid
import torch.nn.functional as F
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from Evaluate.MI import MI_function
from Evaluate.VIF import vifp_mscale
from Evaluate.niqe import niqe
from Evaluate.simple_metric import *
import matplotlib.colors
import re
def count_parameters(model):
    """
    Calculate the total number of trainable parameters in the model
    
    Args:
        model: Neural network model
    
    Returns:
        int: Total number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_images(save_path, filename, images, subdirs):
    """
    Save multiple images to specified subdirectories
    
    Args:
        save_path: Base path for saving images
        filename: Name of the file to save
        images: List of images to save
        subdirs: List of subdirectories corresponding to each image
    
    Returns:
        bool: True if all images were saved successfully, False otherwise
    """
    if len(images) != len(subdirs):
        print("Number of images does not match number of subdirectories")
        return False

    for img, subdir in zip(images, subdirs):
        path = os.path.join(save_path, subdir, filename)
        if img is None:
            print(f"Image is None for {subdir}/{filename}")
            return False

        # Check if image is already in uint8 format
        if img.dtype == np.uint8:
            save_img = img
        else:
            # Check the range of the image
            img_min, img_max = img.min(), img.max()

            if img_max <= 1.0 and img_min >= 0.0:
                # Image is in [0, 1] range, scale to [0, 255]
                save_img = (img * 255).astype(np.uint8)
            elif img_max <= 255 and img_min >= 0:
                # Image is already in [0, 255] range, just convert to uint8
                save_img = img.astype(np.uint8)
            else:
                # Image is in an unknown range, normalize to [0, 255]
                save_img = ((img - img_min) / (img_max - img_min) * 255).astype(np.uint8)
                print(
                    f"Image {subdir}/{filename} had an unexpected range [{img_min}, {img_max}]. Normalized to [0, 255].")

        if not cv2.imwrite(path, save_img):
            print(f"Failed to save image: {path}")
            return False

    return True

def calculate_metrics(fused_image, all_in_focus_gt, estimated_depth, input_depth_map):
    """
    Calculate various evaluation metrics for image fusion and depth estimation
    
    Args:
        fused_image: The fused image
        all_in_focus_gt: All-in-focus ground truth image
        estimated_depth: Estimated depth map
        input_depth_map: Input depth map ground truth
    
    Returns:
        dict: Dictionary containing various evaluation metrics
    """
    metrics = {}

    if all_in_focus_gt is not None:
        # 计算与全清晰图像相关的指标
        metrics['SSIM'] = compare_ssim(fused_image, all_in_focus_gt)
        metrics['PSNR'] = compare_psnr(fused_image, all_in_focus_gt)
        metrics['MSE'] = MSE_function(fused_image, all_in_focus_gt)
        metrics['MAE'] = MAE_function(fused_image, all_in_focus_gt)
        metrics['RMSE'] = RMSE_function(fused_image, all_in_focus_gt)
        metrics['logRMS'] = logRMS_function(fused_image, all_in_focus_gt)
        metrics['abs_rel_error'] = abs_rel_error_function(fused_image, all_in_focus_gt)
        metrics['sqr_rel_error'] = sqr_rel_error_function(fused_image, all_in_focus_gt)
        metrics['VIF'] = vifp_mscale(fused_image, all_in_focus_gt)
        metrics['MI'] = MI_function(fused_image, all_in_focus_gt)
        metrics['NIQE'] = niqe(fused_image)
        metrics['SF'] = SF_function(fused_image)
        metrics['AVG'] = AG_function(fused_image)
        metrics['EN'] = EN_function(fused_image)
        metrics['STD'] = SD_function(fused_image)

    if input_depth_map is not None:
        # 计算与深度图相关的指标
        metrics['depth_mse'] = MSE_function(estimated_depth, input_depth_map)
        metrics['depth_mae'] = MAE_function(estimated_depth, input_depth_map)
    return metrics

def to_image(tensor_batch, epoch, tag, path, nrow=6):
    """
    Convert tensor batch to image and save it
    
    Args:
        tensor_batch: Input tensor batch
        epoch: Current training epoch
        tag: Image tag
        path: Save path
        nrow: Number of images per row in the grid, default is 6
    """
    if not os.path.isdir(path):
        os.makedirs(path)

    # Ensure the input is a 4D tensor (batch_size, channels, height, width)
    if tensor_batch.dim() == 3:
        tensor_batch = tensor_batch.unsqueeze(0)

    # Normalize the tensor if it's not in the range [0, 1]
    if tensor_batch.min() < 0 or tensor_batch.max() > 1:
        tensor_batch = (tensor_batch - tensor_batch.min()) / (tensor_batch.max() - tensor_batch.min())

    # Create a grid of images
    grid = make_grid(tensor_batch, nrow=nrow, padding=2, normalize=True)

    # Save the grid as an image
    save_image(grid, os.path.join(path, f'{epoch}_{tag}.jpg'))

def resize_to_multiple_of_32(image):
    """
    Resize image to be multiple of 32 in both dimensions
    
    Args:
        image: Input image tensor
    
    Returns:
        tuple: (Resized image, Original image dimensions)
    """
    h, w = image.shape[-2:]
    new_h = ((h - 1) // 32 + 1) * 32
    new_w = ((w - 1) // 32 + 1) * 32
    resized_image = F.interpolate(image, size=(new_h, new_w), mode='bilinear', align_corners=False)
    return resized_image, (h, w)


def gray_to_colormap(img, cmap='rainbow'):
    """
    Convert grayscale image to colormap
    
    Args:
        img: Input grayscale image, must be 2D array
        cmap: Matplotlib colormap name, default is 'rainbow'
    
    Returns:
        ndarray: Converted colormap image in uint8 format
    
    Note:
        - Input image should be 2D
        - Negative values will be set to 0
        - Values less than 1e-10 will be masked as invalid
    """
    assert img.ndim == 2

    img[img < 0] = 0
    mask_invalid = img < 1e-10
    img = img / (img.max() + 1e-8)
    norm = matplotlib.colors.Normalize(vmin=0, vmax=1.1)
    cmap_m = matplotlib.cm.get_cmap(cmap)
    map = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap_m)
    colormap = (map.to_rgba(img)[:, :, :3] * 255).astype(np.uint8)
    colormap[mask_invalid] = 0
    return colormap

def config_model_dir(resume=False, subdir_name='train_runs'):
    """
    Configure and create model directory for saving training results
    
    Args:
        resume: Boolean flag indicating whether to resume training from existing directory
        subdir_name: Base name for the subdirectory, default is 'train_runs'
    
    Returns:
        str: Path to the model directory
        
    Note:
        - Creates a new numbered directory if resume=False
        - Returns existing directory path if resume=True
        - Directory naming format: {subdir_name}{number} (e.g., train_runs1, train_runs2)
    """
    # Get current project directory
    project_dir = os.getcwd()
    # Get path to models directory
    models_dir = os.path.join(project_dir, subdir_name)
    # Create models directory if it doesn't exist
    if not os.path.exists(models_dir):
        os.mkdir(models_dir)

    # Create first directory if none exists
    if not os.path.exists(os.path.join(models_dir, subdir_name+'1')):
        os.mkdir(os.path.join(models_dir, subdir_name+'1'))
        return os.path.join(models_dir, subdir_name+'1')
    else:
        # Get existing subdirectories
        sub_dirs = [d for d in os.listdir(models_dir) if os.path.isdir(os.path.join(models_dir, d))]
        sub_dirs.sort(key=lambda l: int(re.findall('\d+', l)[0]))
        last_numbers = re.findall("\d+", sub_dirs[-1])  # list

        if not resume:
            # Create new directory with incremented number
            new_sub_dir_name = subdir_name + str(int(last_numbers[0]) + 1)
        else:
            # Use existing directory for resume
            new_sub_dir_name = subdir_name + str(int(last_numbers[0]))

        model_dir_path = os.path.join(models_dir, new_sub_dir_name)
        
        if not resume:
            os.mkdir(model_dir_path)
            
        print(model_dir_path)
        return model_dir_path
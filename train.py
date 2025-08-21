# -*- coding: utf-8 -*-
# @Author  : XinZhe Xie
# @University  : ZheJiang University

"""
Main training script.
Supports multiple datasets and validation sets.
"""

import argparse
import time
from torch.optim import AdamW
from torch.optim.lr_scheduler import ExponentialLR
from torchmetrics import MeanAbsoluteError, MeanSquaredError
from tqdm import tqdm
from Dataloader import get_updated_dataloader
from network import StackMFF_V2
import torch
import os
import torch.nn as nn
import pandas as pd
from utils import to_image, count_parameters, config_model_dir

def parse_args():
    """
    Parse command line arguments for the training script.
    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(description="Training script for depth estimation")
    
    # Model save name
    parser.add_argument('--save_name', default='train_runs', help='Name for saving the model and logs')

    # Dataset paths for training
    parser.add_argument('--train_stack', default='', type=str, help='Path to first training stack directory')
    parser.add_argument('--train_depth_continuous', default='', type=str, help='Path to first training depth maps')
    parser.add_argument('--train_stack_2', default='', type=str, help='Path to second training stack directory')
    parser.add_argument('--train_depth_continuous_2', default='', type=str, help='Path to second training depth maps')
    parser.add_argument('--train_stack_3', default='', type=str, help='Path to third training stack directory')
    parser.add_argument('--train_depth_continuous_3', default='', type=str, help='Path to third training depth maps')
    parser.add_argument('--train_stack_4', default='', type=str, help='Path to fourth training stack directory')
    parser.add_argument('--train_depth_continuous_4', default='', type=str, help='Path to fourth training depth maps')
    parser.add_argument('--train_stack_5', default='', type=str, help='Path to fifth training stack directory')
    parser.add_argument('--train_depth_continuous_5', default='', type=str, help='Path to fifth training depth maps')

    # Dataset paths for validation
    parser.add_argument('--val_stack', default='', type=str, help='Path to first validation stack directory')
    parser.add_argument('--val_depth_continuous', default='', type=str, help='Path to first validation depth maps')
    parser.add_argument('--val_stack_2', default='', type=str, help='Path to second validation stack directory')
    parser.add_argument('--val_depth_continuous_2', default='', type=str, help='Path to second validation depth maps')
    parser.add_argument('--val_stack_3', default='', type=str, help='Path to third validation stack directory')
    parser.add_argument('--val_depth_continuous_3', default='', type=str, help='Path to third validation depth maps')
    parser.add_argument('--val_stack_4', default='', type=str, help='Path to fourth validation stack directory')
    parser.add_argument('--val_depth_continuous_4', default='', type=str, help='Path to fourth validation depth maps')
    parser.add_argument('--val_stack_5', default='', type=str, help='Path to fifth validation stack directory')
    parser.add_argument('--val_depth_continuous_5', default='', type=str, help='Path to fifth validation depth maps')

    # Dataset usage flags
    parser.add_argument('--use_train_dataset_1', type=bool, default=True, help='Whether to use first training dataset')
    parser.add_argument('--use_train_dataset_2', type=bool, default=True, help='Whether to use second training dataset')
    parser.add_argument('--use_train_dataset_3', type=bool, default=True, help='Whether to use third training dataset')
    parser.add_argument('--use_train_dataset_4', type=bool, default=True, help='Whether to use fourth training dataset')
    parser.add_argument('--use_train_dataset_5', type=bool, default=True, help='Whether to use fifth training dataset')
    parser.add_argument('--use_val_dataset_1', type=bool, default=True, help='Whether to use first validation dataset')
    parser.add_argument('--use_val_dataset_2', type=bool, default=True, help='Whether to use second validation dataset')
    parser.add_argument('--use_val_dataset_3', type=bool, default=True, help='Whether to use third validation dataset')
    parser.add_argument('--use_val_dataset_4', type=bool, default=True, help='Whether to use fourth validation dataset')
    parser.add_argument('--use_val_dataset_5', type=bool, default=True, help='Whether to use fifth validation dataset')

    # Training parameters
    parser.add_argument('--subset_fraction_train', type=float, default=1, help='Fraction of training data to use')
    parser.add_argument('--subset_fraction_val', type=float, default=0.2, help='Fraction of validation data to use')
    parser.add_argument('--training_image_size', type=int, default=384, help='Size of training images')
    parser.add_argument('--batch_size', type=int, default=12, help='Batch size for training and validation')
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--eval_interval', type=int, default=10, help='Interval between evaluations')
    parser.add_argument('--lr', type=float, default=1e-3, help='Initial learning rate')
    parser.add_argument('--lr_decay', type=float, default=0.9, help='Learning rate decay factor')
    parser.add_argument('--loss_ratio', type=list, default=[0,1], help='Ratio between different loss components')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of data loading workers')

    return parser.parse_args()

def train(model, train_loader, criterion_depth, optimizer, device, loss_ratio, epoch):
    """
    Training function for one epoch.
    
    Args:
        model: Neural network model
        train_loader: DataLoader for training data
        criterion_depth: Loss function for depth estimation
        optimizer: Optimization algorithm
        device: Device to run the training on
        loss_ratio: Weights for different loss components
        epoch: Current epoch number
    
    Returns:
        tuple: Average training loss and depth loss
    """
    model.train()
    train_loss = 0.0
    loss_depth_total = 0.0
    progress_bar = tqdm(train_loader, desc="Training")

    for image_stack, depth_map_gt, stack_size in progress_bar:
        image_stack, depth_map_gt = image_stack.to(device), depth_map_gt.to(device)

        optimizer.zero_grad()
        _, depth_map, _ = model(image_stack)

        loss_depth = criterion_depth(depth_map, depth_map_gt)
        total_loss = loss_ratio[1] * loss_depth

        total_loss.backward()
        optimizer.step()

        train_loss += total_loss.item()
        loss_depth_total += loss_depth.item()

        progress_bar.set_postfix({
            "Epoch": f"{epoch}",
            "total_loss": f"{total_loss.item():.6f}",
            "loss_depth": f"{loss_depth.item():.6f}",
        })

    return (train_loss / len(train_loader),
            loss_depth_total / len(train_loader))

def validate_dataset(model, val_loader, criterion_depth, device, epoch, save_path, loss_ratio):
    """
    Validation function for one dataset.
    
    Args:
        model: Neural network model
        val_loader: DataLoader for validation data
        criterion_depth: Loss function for depth estimation
        device: Device to run the validation on
        epoch: Current epoch number
        save_path: Path to save validation results
        loss_ratio: Weights for different loss components
    
    Returns:
        tuple: Validation metrics (loss, depth loss, MSE, MAE)
    """
    model.eval()
    val_loss = 0.0
    loss_depth_total = 0.0

    depth_mse_metric = MeanSquaredError().to(device)
    depth_mae_metric = MeanAbsoluteError().to(device)

    total_depth_mse = 0.0
    total_depth_mae = 0.0

    progress_bar = tqdm(val_loader, desc=f"Validating Epoch {epoch}")

    with torch.no_grad():
        for i, (image_stack, depth_map_gt, stack_size) in enumerate(progress_bar):
            image_stack, depth_map_gt = image_stack.to(device), depth_map_gt.to(device)

            _, depth_map, _ = model(image_stack)

            loss_depth = criterion_depth(depth_map, depth_map_gt)
            total_loss = loss_ratio[1] * loss_depth

            val_loss += total_loss.item()
            loss_depth_total += loss_depth.item()

            depth_mse = depth_mse_metric(depth_map, depth_map_gt)
            depth_mae = depth_mae_metric(depth_map, depth_map_gt)

            total_depth_mse += depth_mse.item()
            total_depth_mae += depth_mae.item()

            progress_bar.set_postfix({
                "Total Loss": f"{total_loss.item():.6f}",
                "Depth MSE": f"{depth_mse.item():.6f}",
                "Depth MAE": f"{depth_mae.item():.6f}"
            })

            if i == len(val_loader) - 1:
                visualization_path = os.path.join(save_path, f'validation_visualization/epoch_{epoch}')
                to_image(depth_map_gt, epoch, 'depth_map_gt', visualization_path)
                to_image(depth_map, epoch, 'depth_map', visualization_path)

    num_batches = len(val_loader)
    avg_depth_mse = total_depth_mse / num_batches
    avg_depth_mae = total_depth_mae / num_batches

    return (val_loss / num_batches,
            loss_depth_total / num_batches,
            avg_depth_mse, avg_depth_mae)

def main():
    """
    Main training function that handles:
    - Argument parsing
    - Dataset loading
    - Model initialization
    - Training loop
    - Validation
    - Metrics logging
    - Model saving
    """
    args = parse_args()
    model_save_path = config_model_dir(resume=False, subdir_name=args.save_name)

    # Training data setup
    train_dataset_params = []
    if args.use_train_dataset_1:
        train_dataset_params.append({
            'root_dir': args.train_stack,
            'continuous_depth_dir': args.train_depth_continuous,
            'subset_fraction': args.subset_fraction_train
        })
    if args.use_train_dataset_2:
        train_dataset_params.append({
            'root_dir': args.train_stack_2,
            'continuous_depth_dir': args.train_depth_continuous_2,
            'subset_fraction': args.subset_fraction_train
        })
    if args.use_train_dataset_3:
        train_dataset_params.append({
            'root_dir': args.train_stack_3,
            'continuous_depth_dir': args.train_depth_continuous_3,
            'subset_fraction': args.subset_fraction_train
        })
    if args.use_train_dataset_4:
        train_dataset_params.append({
            'root_dir': args.train_stack_4,
            'continuous_depth_dir': args.train_depth_continuous_4,
            'subset_fraction': args.subset_fraction_train
        })
    if args.use_train_dataset_5:
        train_dataset_params.append({
            'root_dir': args.train_stack_5,
            'continuous_depth_dir': args.train_depth_continuous_5,
            'subset_fraction': args.subset_fraction_train
        })
    train_loader = get_updated_dataloader(
        train_dataset_params,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        augment=True,
        target_size=args.training_image_size
    )

    # Separate validation loaders for each dataset
    val_loaders = []
    if args.use_val_dataset_1:
        val_loader_1 = get_updated_dataloader(
            [{
                'root_dir': args.val_stack,
                'continuous_depth_dir': args.val_depth_continuous,
                'subset_fraction': args.subset_fraction_val
            }],
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            augment=False,
            target_size=args.training_image_size
        )
        val_loaders.append(val_loader_1)

    if args.use_val_dataset_2:
        val_loader_2 = get_updated_dataloader(
            [{
                'root_dir': args.val_stack_2,
                'continuous_depth_dir': args.val_depth_continuous_2,
                'subset_fraction': args.subset_fraction_val
            }],
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            augment=False,
            target_size=args.training_image_size
        )
        val_loaders.append(val_loader_2)

    if args.use_val_dataset_3:
        val_loader_3 = get_updated_dataloader(
            [{
                'root_dir': args.val_stack_3,
                'continuous_depth_dir': args.val_depth_continuous_3,
                'subset_fraction': args.subset_fraction_val
            }],
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            augment=False,
            target_size=args.training_image_size
        )
        val_loaders.append(val_loader_3)

    if args.use_val_dataset_4:
        val_loader_4 = get_updated_dataloader(
            [{
                'root_dir': args.val_stack_4,
                'continuous_depth_dir': args.val_depth_continuous_4,
                'subset_fraction': args.subset_fraction_val
            }],
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            augment=False,
            target_size=args.training_image_size
        )
        val_loaders.append(val_loader_4)

    if args.use_val_dataset_5:
        val_loader_5 = get_updated_dataloader(
            [{
                'root_dir': args.val_stack_5,
                'continuous_depth_dir': args.val_depth_continuous_5,
                'subset_fraction': args.subset_fraction_val
            }],
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            augment=False,
            target_size=args.training_image_size
        )
        val_loaders.append(val_loader_5)

    print(f"Training samples: {len(train_loader.dataset) if train_loader else 0}")
    for i, val_loader in enumerate(val_loaders, 1):
        print(f"Validation samples (Dataset {i}): {len(val_loader.dataset)}")

    # Model setup
    model = StackMFF_V2()
    num_params = count_parameters(model)
    print(f"Number of trainable parameters: {num_params}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model = nn.DataParallel(model)

    # Loss functions, optimizer, and scheduler
    criterion_depth = torch.nn.MSELoss()
    optimizer = AdamW(model.parameters(), lr=args.lr)
    scheduler = ExponentialLR(optimizer, gamma=args.lr_decay)

    best_val_loss = float('inf')
    start_time = time.time()
    val_results_data = []

    for epoch in range(args.num_epochs):
        print(f"Epoch {epoch + 1}/{args.num_epochs}")

        # Training
        if train_loader:
            train_loss, train_depth_loss = train(model, train_loader, criterion_depth, optimizer, device, args.loss_ratio, epoch)

        # Validation
        val_results = []
        epoch_val_data = {'epoch': epoch + 1}
        for i, val_loader in enumerate(val_loaders, 1):
            results = validate_dataset(model, val_loader, criterion_depth, device, epoch, 
                                    os.path.join(model_save_path, f'val_dataset_{i}'), args.loss_ratio)
            val_results.append(results)
            (val_loss, val_depth_loss, avg_depth_mse, avg_depth_mae) = results

            epoch_val_data.update({
                f'val_dataset_{i}_loss': val_loss,
                f'val_dataset_{i}_depth_loss': val_depth_loss,
                f'val_dataset_{i}_depth_mse': avg_depth_mse,
                f'val_dataset_{i}_depth_mae': avg_depth_mae
            })

            print(f"Validation Dataset {i} Results:")
            print(f"  Loss: {val_loss:.6f}")
            print(f"  Depth MSE: {avg_depth_mse:.6f}")
            print(f"  Depth MAE: {avg_depth_mae:.6f}")

        # Add training loss to the epoch data
        if train_loader:
            epoch_val_data.update({
                'train_loss': train_loss,
                'learning_rate': scheduler.get_last_lr()[0]
            })

        # Save validation results
        val_results_data.append(epoch_val_data)
        val_results_df = pd.DataFrame(val_results_data)
        val_results_df.to_csv(os.path.join(model_save_path, 'validation_results.csv'), index=False)

        # Save model
        os.makedirs(os.path.join(model_save_path, 'model_save'), exist_ok=True)
        torch.save(model.state_dict(), f"{os.path.join(model_save_path, 'model_save')}/epoch_{epoch}.pth")

        # Check for best model
        if val_loaders:
            avg_val_loss = sum(results[0] for results in val_results) / len(val_results)
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), f"{model_save_path}/best_model.pth")
                print(f"Saved new best model with validation loss: {best_val_loss:.6f}")

        scheduler.step()

    end_time = time.time()
    training_time_hours = (end_time - start_time) / 3600
    print(f"Training completed in {training_time_hours:.2f} hours")
    print(f"Model saved at: {model_save_path}")

if __name__ == "__main__":

    main()

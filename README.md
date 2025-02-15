# Rethinking Multi-focus Image Stack Fusion: A Lightweight One-shot Deep Learning Framework via Focal Plane Depth Regression (StackMFF-V2)

<div align="center">

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.8+-ee4c2c.svg)](https://pytorch.org/get-started/locally/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

</div>

## 📝 Table of Contents

- [Overview](#-overview)
- [Highlights](#-highlights)
- [Installation](#-installation)
- [Data Preparation](#-data-preparation)
- [Usage](#-usage)
- [Results](#-results)
- [Citation](#-citation)
- [License](#-license)

## 📖 Overview

Most multi-focus image fusion (MFF) networks are designed for two-image fusion, requiring multiple iterative operations for stack fusion that lead to error accumulation and image quality degradation. To address this challenge, we rethink the multi-focus image stack fusion problem by treating image stacks as a unified entity and propose a lightweight one-shot deep learning framework based on focal plane depth regression. The framework consists of three stages: intra-layer focus estimation, inter-layer focus estimation, and focus map regression. By reformulating multi-focus image stack fusion as a focal plane depth regression task, our framework enables end-to-end training using depth maps as proxy supervision. Extensive experiments on five public datasets demonstrate that our framework achieves state-of-the-art performance while reducing model size by 99.2% (from 6.08M to 0.05M parameters) compared to our previous one-shot fusion framework StackMFF, with the lowest FLOPs growth rate as the number of input images increases. Furthermore, our framework can process image stacks of arbitrary size in a single operation while preserving pixel fidelity through direct sampling.

<div align="center">
<img src="assets/framework.png" width="800px"/>
<p>Overview of StackMFF-V2 Framework</p>
</div>

## ✨ Highlights

- 🔄 A novel framework for multi-focus image stack fusion based on focal plane regression
- 🎯 Leverage depth maps as proxy supervision signals for focus map regression
- 📊 Process stacks of any size while preserving pixel fidelity through direct sampling
- ⚡ Achieve SOTA performance with minimal model size and computational cost

## 🚀 Installation

1. Clone the repository:
```bash
git clone https://github.com/Xinzhe99/StackMFF-V2.git
cd StackMFF-V2
```

2. Create and activate a virtual environment (recommended):
```bash
conda create -n stackmffv2 python=3.8
conda activate stackmffv2
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## �� Data Preparation

### Training Data Structure

The training dataset should be organized in the following structure:

```
data/
├── train/
│   ├── stack1/
│   │   ├── 1.png
│   │   ├── 2.png
│   │   └── ...
│   └── stack2/
│       ├── 1.png
│       └── ...
├── val/
└── test/
```

Depth maps should be stored in corresponding directories:

```
depth_maps/
├── train/
│   ├── stack1.png
│   └── stack2.png
└── val/
```

### Test Data Structure

For batch testing multiple datasets, organize your test data as follows:

```
test_root/
├── dataset1/
│   └── dof_stack/
│       ├── scene1/
│       │   ├── 1.png
│       │   ├── 2.png
│       │   └── ...
│       └── scene2/
│           ├── 1.png
│           ├── 2.png
│           └── ...
├── dataset2/
│   └── dof_stack/
│       ├── scene1/
│       └── scene2/
└── dataset3/
    └── dof_stack/
        ├── scene1/
        └── scene2/
```

Each dataset folder (e.g., Mobile_Depth, Middlebury, FlyingThings3D) should contain a `dof_stack` subfolder with multiple scene folders. Each scene folder contains the multi-focus image stack numbered sequentially.

## 💻 Usage

The pre-trained model weights file `model.pth` should be placed in the project root directory.

### Predict Single Stack

```bash
python predict_one_stack.py \
    --model_path model.pth \
    --input_dir path/to/input/stack \
    --output_dir path/to/output
```

### Predict Dataset

```bash
python predict_datasets.py \
    --model_path model.pth \
    --test_root path/to/test/root \
    --test_datasets dataset1 dataset2 \
    --output_dir results
```

Parameters:
- `--input_dir`: Directory containing input image stack
- `--output_dir`: Directory for saving results
- `--model_path`: Path to model weights file (optional, defaults to `model.pth` in root directory)
- `--test_root`: Root directory of test datasets
- `--test_datasets`: List of dataset names to test

### Training

```bash
python train.py \
    --model_path model.pth \
    --train_data_path data/train \
    --train_depth_path depth_maps/train \
    --val_data_path data/val \
    --val_depth_path depth_maps/val \
    --batch_size 8 \
    --epochs 100 \
    --lr 0.0001
```

Training Parameters:
- `--train_data_path`: Path to training dataset
- `--train_depth_path`: Path to training depth maps
- `--model_path`: Path to model weights file
- `--batch_size`: Batch size for training
- `--epochs`: Number of training epochs
- `--lr`: Learning rate

## 🎯 Results

### Model Complexity Analysis

| Method | Model Size (M) | FLOPs (G) | Number of Runs |
|--------|---------------|------------|----------------|
| IFCNN | 0.08 | 8.54 | N-1 |
| U2Fusion | 0.66 | 86.4 | N-1 |
| SDNet | *0.07* | 8.81 | N-1 |
| MFF-GAN | **0.05** | *3.08* | N-1 |
| SwinFusion | 0.93 | 63.73 | N-1 |
| MUFusion | 2.16 | 24.07 | N-1 |
| SwinMFF | 41.25 | 22.38 | N-1 |
| DDBFusion | 10.92 | 184.93 | N-1 |
| StackMFF | 6.08 | 21.98 | 1 |
| **Proposed** | **0.05** | **2.75** | 1 |
| Reduction (%) | 28.57% | 10.71% | - |

*Note: N represents the number of images in the stack. Model Size is in millions of parameters, FLOPs is in billions for fusing two images.*

### Quantitative Evaluation on Public Datasets

| Method | Mobile Depth | Middlebury | FlyingThings3D | Road-MF | NYU Depth V2 |
|--------|--------------|------------|----------------|----------|---------------|
|  | SSIM↑/PSNR↑ | SSIM↑/PSNR↑ | SSIM↑/PSNR↑ | SSIM↑/PSNR↑ | SSIM↑/PSNR↑ |
| CVT | 0.9368/32.6158 | 0.8893/29.3426 | 0.9157/30.0917 | 0.9777/36.0578 | 0.9717/38.8186 |
| DWT | 0.9340/32.1651 | 0.8850/29.1761 | 0.9123/30.0074 | 0.9309/30.3456 | 0.9594/35.8626 |
| DCT | 0.4720/17.2719 | 0.4520/13.9972 | 0.4603/15.0949 | 0.4856/16.9598 | 0.4802/14.2216 |
| DTCWT | 0.9412/32.7641 | 0.8938/29.3763 | 0.9203/30.1512 | 0.9826/36.7138 | 0.9743/39.1475 |
| NSCT | 0.9340/32.1651 | 0.8850/29.1761 | 0.9123/30.0074 | 0.9813/37.0137 | 0.9707/38.7653 |
| IFCNN | 0.7882/24.9863 | 0.9014/29.2064 | 0.9236/31.3069 | 0.8952/27.6907 | 0.9364/34.3915 |
| U2Fusion | 0.3788/10.0482 | 0.3980/10.1318 | 0.4242/11.4382 | 0.3811/10.8764 | 0.3869/10.7027 |
| SDNet | 0.3961/12.1659 | 0.4399/14.0048 | 0.4457/14.5929 | 0.4144/13.0182 | 0.4212/14.2688 |
| MFF-GAN | 0.1797/7.1264 | 0.2962/10.1180 | 0.3006/11.9173 | 0.2559/9.3437 | 0.2755/10.5829 |
| SwinFusion | 0.4381/12.4597 | 0.4254/13.4794 | 0.4313/14.1286 | 0.3945/11.9315 | 0.4114/13.6265 |
| MUFusion | 0.4819/18.7311 | 0.5809/19.7779 | 0.4762/19.8073 | 0.6821/19.6156 | 0.5891/21.0372 |
| SwinMFF | 0.3511/10.8676 | 0.4215/11.8564 | 0.3238/12.2809 | 0.4795/13.2869 | 0.3983/13.1620 |
| DDBFusion | 0.8365/26.3713 | 0.7181/23.7650 | 0.6984/23.0223 | 0.8065/24.4036 | 0.8786/28.7440 |
| StackMFF | 0.4768/16.3399 | 0.4642/15.5382 | 0.4741/16.2531 | 0.4846/16.5069 | 0.4906/18.8776 |
| **Proposed** | **0.9452**/**34.4852** | **0.9123**/**30.0263** | **0.9405**/**32.0477** | 0.9610/32.7955 | **0.9823**/38.5155 |

### Computational Efficiency (seconds)

| Method | Device | Mobile Depth | Middlebury | FlyingThings3D | Road-MF | NYU Depth V2 |
|--------|---------|--------------|------------|----------------|----------|---------------|
| CVT | CPU | 48.00 | 31.37 | 37.87 | 78.14 | 56.20 |
| DWT | CPU | 5.34 | 8.62 | 6.75 | 4.62 | 3.45 |
| DCT | CPU | 4.97 | 3.30 | 6.04 | 4.97 | 9.16 |
| DTCWT | CPU | 11.44 | 9.40 | 14.70 | 12.82 | 10.06 |
| NSCT | CPU | 231.84 | 165.13 | 133.84 | 217.03 | 152.05 |
| IFCNN | GPU | 0.55 | 0.50 | 0.78 | 0.55 | 0.44 |
| U2Fusion | CPU | 41.04 | 35.90 | 45.10 | 104.96 | 41.93 |
| SDNet | CPU | 9.68 | 5.26 | 14.04 | 8.18 | 6.64 |
| MFF-GAN | CPU | 6.40 | 8.88 | 10.06 | 12.67 | 6.98 |
| SwinFusion | GPU | 28.21 | 19.53 | 32.33 | 30.19 | 67.52 |
| MUFusion | GPU | 40.40 | 21.98 | 55.02 | 45.79 | 31.02 |
| SwinMFF | GPU | 27.97 | 18.23 | 34.05 | 55.04 | 24.47 |
| DDBFusion | GPU | 33.89 | 30.06 | 41.98 | 35.57 | 17.35 |
| StackMFF | GPU | 0.22 | 0.19 | 0.24 | 0.22 | 0.20 |
| **Proposed** | GPU | **0.14** | **0.08** | **0.11** | **0.11** | **0.07** |

### Statistical Ranking Analysis

| Method | Mobile Depth | Middlebury | FlyingThings3D | Road-MF | Overall |
|--------|--------------|------------|----------------|----------|----------|
| MFF-GAN | 15.0 | 15.0 | 14.0 | 15.0 | 14.75 |
| U2Fusion | 14.0 | 14.0 | 13.5 | 14.0 | 13.88 |
| SwinFusion | 11.0 | 13.0 | 12.0 | 13.0 | 12.25 |
| SDNet | 12.5 | 11.0 | 12.0 | 12.0 | 11.88 |
| DCT | 11.0 | 12.0 | 11.0 | 11.0 | 11.25 |
| SwinMFF | 11.0 | 10.0 | 9.5 | 10.0 | 10.13 |
| MUFusion | 9.0 | 8.0 | 8.5 | 8.0 | 8.38 |
| StackMFF | 8.5 | 9.0 | 6.5 | 9.0 | 8.25 |
| DDBFusion | 6.0 | 7.0 | 7.5 | 5.0 | 6.38 |
| DWT | 4.5 | 5.0 | 5.0 | 7.0 | 5.38 |
| IFCNN | 7.0 | 3.0 | 2.0 | 6.0 | 4.50 |
| NSCT | 4.5 | 5.0 | 5.0 | 1.5 | 4.00 |
| CVT | 3.5 | 3.5 | 4.0 | 3.0 | 3.50 |
| DTCWT | 2.0 | 2.5 | 3.0 | 1.5 | 2.25 |
| **Proposed** | **1.0** | **1.0** | **1.0** | 4.0 | **1.75** |

## 📚 Citation

If you use this project in your research, please cite our paper:

```bibtex
@article{xie2024swinmff,
  title={SwinMFF: toward high-fidelity end-to-end multi-focus image fusion via swin transformer-based network},
  author={Xie, Xinzhe and Guo, Buyu and Li, Peiliang and He, Shuangyan and Zhou, Sangjun},
  journal={The Visual Computer},
  pages={1--24},
  year={2024},
  publisher={Springer}
}
```

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## 📧 Contact

- Author: XinZhe Xie
- Institution: Zhejiang University
- Email: [xiexinzhe@zju.edu.cn]

---

<div align="center">
⭐ If you find this project helpful, please give it a star!
</div> 

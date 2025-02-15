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

## 📂 Data Preparation

The dataset should be organized in the following structure:

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

## 💻 Usage

### Training

```bash
python train.py \
    --train_data_path data/train \
    --train_depth_path depth_maps/train \
    --val_data_path data/val \
    --val_depth_path depth_maps/val \
    --batch_size 8 \
    --epochs 100 \
    --lr 0.0001
```

Key Parameters:
- `--train_data_path`: Path to training dataset
- `--train_depth_path`: Path to training depth maps
- `--batch_size`: Batch size
- `--epochs`: Number of training epochs
- `--lr`: Learning rate

### Predicting Single Stack

```bash
python predict_one_stack.py \
    --model_path checkpoints/model.pth \
    --input_dir path/to/input/stack \
    --output_dir path/to/output
```

### Predicting Dataset

```bash
python predict_datasets.py \
    --model_path checkpoints/model.pth \
    --test_data_path data/test \
    --output_dir results
```

### Evaluation

```bash
python evaluate.py \
    --model_path checkpoints/model.pth \
    --test_data_path data/test \
    --metrics "PSNR SSIM VIF NIQE"
```

## 📊 Results

### Quantitative Evaluation on Public Datasets

| Dataset | Method | SSIM↑ | PSNR↑ |
|---------|---------|-------|--------|
| Mobile Depth | StackMFF | 0.4768 | 16.3399 |
| | **Proposed** | **0.9452** | **34.4852** |
| Middlebury | StackMFF | 0.4642 | 15.5382 |
| | **Proposed** | **0.9123** | **30.0263** |
| FlyingThings3D | StackMFF | 0.4741 | 16.2531 |
| | **Proposed** | **0.9405** | **32.0477** |
| Road-MF | StackMFF | 0.4846 | 16.5069 |
| | **Proposed** | **0.9610** | **32.7955** |
| NYU Depth V2 | StackMFF | 0.4906 | 18.8776 |
| | **Proposed** | **0.9823** | **38.5155** |

### Computational Efficiency (seconds)

| Method | Mobile Depth | Middlebury | FlyingThings3D | Road-MF | NYU Depth V2 |
|--------|--------------|------------|----------------|----------|---------------|
| StackMFF | 0.22 | 0.19 | 0.24 | 0.22 | 0.20 |
| **Proposed** | **0.14** | **0.08** | **0.11** | **0.11** | **0.07** |
| Reduction (%) | 36.36% | 57.89% | 54.17% | 50.00% | 65.00% |

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

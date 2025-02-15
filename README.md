# Rethinking Multi-focus Image Stack Fusion: A Lightweight One-shot Deep Learning Framework via Focal Plane Depth Regression (StackMFF-V2)

<div align="center">

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.8+-ee4c2c.svg)](https://pytorch.org/get-started/locally/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

</div>

## 📝 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Installation](#-installation)
- [Data Preparation](#-data-preparation)
- [Usage](#-usage)
- [Model Architecture](#-model-architecture)
- [Results](#-results)
- [Citation](#-citation)
- [License](#-license)

## 📖 Overview

StackMFF-V2 is a deep learning-based multi-focus image fusion framework that automatically processes image stacks with different focus areas to generate an all-in-focus image while estimating the scene's focus map. This project is an improved version of the original StackMFF.

<div align="center">
<img src="assets/framework.png" width="800px"/>
<p>Overview of StackMFF-V2 Framework</p>
</div>

## ✨ Features

- 🔄 Support for arbitrary number of input image stacks
- 🎯 High-quality all-in-focus image generation
- 📊 Accurate focus map estimation

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

## 🔍 Model Architecture

StackMFF-V2 consists of three main modules:

1. **Intra-layer focus estimation** (`network.py`)

3. **Inter-layer focus estimation** (`network.py`)

4. **Focus map regression** (`network.py`)

<div align="center">
<img src="assets/attention_module.png" width="600px"/>
<p>Attention Module Architecture</p>
</div>

## 📊 Results

### Quantitative Evaluation

| Method | PSNR↑ | SSIM↑ | VIF↑ | NIQE↓ |
|------|-------|-------|------|-------|
| StackMFF | 32.45 | 0.956 | 0.876 | 3.245 |
| StackMFF-V2 | **33.89** | **0.968** | **0.892** | **2.987** |

### Qualitative Results

<div align="center">
<img src="assets/qualitative_results.png" width="800px"/>
<p>Fusion Results Comparison on Different Scenes</p>
</div>

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

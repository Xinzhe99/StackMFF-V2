<div align="center">

# 📚 StackMFF-V2

**One-Shot Multi-Focus Image Stack Fusion via Focal Depth Regression**

[![Paper](https://img.shields.io/badge/Paper-Applied%20Intelligence-blue)](https://link.springer.com/article/10.1007/s10489-024-06079-8)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.8+-red.svg)](https://pytorch.org/)
[![GitHub](https://img.shields.io/badge/GitHub-StackMFF--V2-black.svg)](https://github.com/Xinzhe99/StackMFF-V2)

*Official PyTorch implementation for One-Shot Multi-Focus Image Stack Fusion via Focal Depth Regression*

</div>

## 📢 News

> [!NOTE]
> 🎉 **2025.08**: We fixed a numerical precision bug in our StackMFF V2 paper's code, which had previously caused degraded fusion image quality.

> 🎉 **2025.08**: We have updated the multifocus image stack registration script `Registration.py` in the code repository. You can now easily integrate it into your own workflow.
> 
> 🎉 **2025.08**: To facilitate user processing of image pair datasets, we provide the `predict_pair_datasets.py` script for batch evaluation of image pair datasets with A/B folder structure. Each dataset is processed separately with organized output folders.

> 🎉 **2025.04**: Our StackMFF V2 paper has been submitted! Coming soon~

> 🎉 **2024.03**: Our StackMFF V1 paper has been accepted by Applied Intelligence (APIN)!

## Authors

**Xinzhe Xie** 👨‍🎓, **Buyu Guo**<sup>✉</sup> 👨‍🏫, **Shuangyan He** 👩‍🏫, **Peiliang Li** 👨‍🏫, **Yanzhen Gu**<sup>✉</sup> 👨‍🏫, **Yanjun Li** 👨‍🏫

### Institutions

🏛️ State Key Laboratory of Ocean Sensing & Ocean College, Zhoushan, P. R. China  
🏛️ Hainan Institute, Zhejiang University, Sanya, P. R. China  
🔬 Donghai Laboratory, Zhoushan, P. R. China
🔬 Hainan Provincial Observatory of Ecological Environment and Fishery Resource in Yazhou Bay, Sanya, P. R. China

<sup>✉</sup> Corresponding author

</div>

##  Table of Contents

- [Overview](#-overview)
- [Highlights](#-highlights)
- [Installation](#-installation)
- [Data Preparation](#-data-preparation)
- [Usage](#-usage)
- [Citation](#-citation)

## 📖 Overview

Multi-focus image fusion is a vital computational imaging technique for applications that require an extended depth of field, including medical imaging, microscopy, professional photography, and autonomous driving. While existing methods excel at fusing image pairs, they often suffer from error accumulation that leads to quality degradation, as well as computational inefficiency when applied to large image stacks. To address these challenges, we introduce a one-shot fusion framework that reframes image-stack fusion as a focal-plane depth regression problem. The framework comprises three key stages: intra-layer focus estimation, inter-layer focus estimation, and focus map regression. By employing a differentiable soft regression strategy and using depth maps as proxy supervisory signals, our method enables end-to-end training without requiring manual focus map annotations. Comprehensive experiments on five public datasets demonstrate that our approach achieves state-of-the-art performance with minimal computational overhead. The resulting efficiency and scalability make the proposed framework a compelling solution for real-time deployment in resource-constrained environments and lay the groundwork for broader practical adoption of multi-focus image fusion.

<div align="center">
<img src="assets/zeromotion_demo.gif" width="800px"/>
</div>


<div align="center">
<img src="assets/framework_new.png" width="800px"/>
<p>Overview of StackMFF-V2 Framework</p>
</div>

## ✨ Highlights

🌟 Reformulates the stack fusion task into a focal plane depth regression problem.

🔑 Depth maps serve as proxy supervision signals, avoiding manual annotations.

🛠️ Employs a differentiable soft-regression strategy to enable end-to-end training.

🎯 Recovers focal depth information during image acquisition via focus map regression.

🏆 Attains SOTA performance with a compact model size and low computational overhead.

 
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

## 📖 Data Preparation

We provide the test datasets used in our paper for research purposes. These datasets were used to evaluate the performance of our proposed method and compare with other state-of-the-art approaches:
- Mobile_Depth
- Middlebury
- FlyingThings3D
- Road_MF
- NYU_Depth_V2

Download Links:
- Test Datasets:
  - Baidu Cloud: [https://pan.baidu.com/s/1vnEciGFDDjDybmoxNSAVSA](https://pan.baidu.com/s/1vnEciGFDDjDybmoxNSAVSA)
  - Extraction Code: cite

- Fusion Results of All Compared Methods:
  - Baidu Cloud: [https://pan.baidu.com/s/1wzv8UKU_0boL1cSs58sr2w](https://pan.baidu.com/s/1wzv8UKU_0boL1cSs58sr2w)
  - Extraction Code: cite

For the implementation of iterative fusion methods mentioned in our paper, please refer to our toolbox:
[Toolbox-for-Multi-focus-Image-Stack-Fusion](https://github.com/Xinzhe99/Toolbox-for-Multi-focus-Image-Stack-Fusion)

These are the exact datasets used in our quantitative evaluation and computational efficiency analysis. After downloading, please organize the datasets following the structure described in the [Predict Dataset](#predict-dataset) section.

The `make_datasets` folder contains all the necessary code for processing and splitting the training datasets:

- `ADE/1_extract.py`: Extracts and organizes images from the ADE20K dataset
- `DUTS/filter.py`: Filters out images with uniform backgrounds from the DUTS dataset
- `DIODE/extract_from_ori.py`: Processes and converts images from the DIODE dataset
- `NYU V2 Depth/`:
  - `1_crop_nyu_v2.py`: Crops RGB and depth images to remove boundary artifacts
  - `2_nyu_depth_norm.py`: Normalizes depth maps to a standard range
  - `3_split.py`: Splits the dataset into training and testing sets
- `Cityscapes/1_move.py`: Reorganizes the Cityscapes dataset into a flattened structure
- `make_dataset.py`: Generates multi-focus image stacks using depth maps

For depth maps, except for the NYU Depth V2 dataset which uses its own depth maps, all other depth maps are obtained through inference using [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2).

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

For batch testing multiple datasets, organize your test data as follows:

```
test_root/
├── Mobile_Depth/
│   └── dof_stack/
│       ├── scene1/
│       │   ├── 1.png
│       │   ├── 2.png
│       │   └── ...
│       └── scene2/
│           ├── 1.png
│           ├── 2.png
│           └── ...
├── Middlebury/
│   └── dof_stack/
│       ├── scene1/
│       └── scene2/
├── FlyingThings3D/
│   └── dof_stack/
├── Road_MF/
│   └── dof_stack/
└── NYU_Depth_V2/
    └── dof_stack/
```

Each dataset folder (e.g., Mobile_Depth, Middlebury, FlyingThings3D, Road_MF, NYU_Depth_V2) should contain a `dof_stack` subfolder with multiple scene folders. Each scene folder contains the multi-focus image stack numbered sequentially.

Run prediction on multiple datasets:
```bash
python predict_datasets.py \
    --model_path model.pth \
    --test_root test_root \
    --test_datasets Mobile_Depth Middlebury FlyingThings3D Road_MF NYU_Depth_V2 \
    --output_dir results
```

The framework will:
1. Test on each dataset independently
2. Generate fusion results for each scene
3. Save results in separate folders for each dataset

Parameters:
- `--test_root`: Root directory containing all test datasets
- `--test_datasets`: List of dataset names to test (e.g., Mobile_Depth Middlebury)
- `--output_dir`: Directory for saving results
- `--model_path`: Path to model weights file (optional, defaults to `model.pth` in root directory)

### Predict Image Pair Datasets

For processing image pair datasets with A/B folder structure, use the `predict_pair_datasets.py` script. This script processes each dataset independently, similar to `predict_datasets.py`, and is specifically designed for datasets where images are organized as pairs in separate 'A' and 'B' subfolders.

Organize your image pair datasets as follows:

```
test_root/
├── dataset1/
│   ├── A/
│   │   ├── 1.png
│   │   ├── 2.png
│   │   └── ...
│   └── B/
│       ├── 1.png
│       ├── 2.png
│       └── ...
├── dataset2/
│   ├── A/
│   └── B/
└── dataset3/
    ├── A/
    └── B/
```

Run prediction on image pair datasets:
```bash
python predict_pair_datasets.py \
    --test_root /path/to/test_root \
    --test_datasets dataset1 dataset2 dataset3 \
    --model_path model.pth \
    --output_dir ./output_pair
```

The script will:
1. Process each dataset independently with separate output folders
2. Automatically match numerically ordered images from A and B folders (e.g., A/1.png pairs with B/1.png)
3. Treat each image pair as a two-image stack for fusion
4. Generate fusion results for each dataset in organized subdirectories
5. Support various image formats (.png, .jpg, .jpeg, .bmp, .tiff, .tif, .webp, .ppm, .pgm, .pbm)

Parameters:
- `--test_root`: Root directory containing image pair datasets
- `--test_datasets`: List of dataset folder names to process
- `--model_path`: Path to model weights file
- `--output_dir`: Directory for saving fusion results (default: ./output_pair)
- `--batch_size`: Batch size for processing (default: 1)
- `--num_workers`: Number of data loading workers (default: 4)

### Training

The framework supports training and validation with multiple datasets. Each dataset should be organized as follows:

```
project_root/
├── train_dataset1/          
│   ├── image_stacks/
│   │   ├── stack1/
│   │   │   ├── 1.png
│   │   │   ├── 2.png
│   │   │   └── ...
│   │   └── stack2/
│   │       ├── 1.png
│   │       ├── 2.png
│   │       └── ...
│   └── depth_maps/
│       ├── stack1.png
│       └── stack2.png
├── train_dataset2/
├── train_dataset3/
├── train_dataset4/
├── train_dataset5/
├── val_dataset1/         
│   ├── image_stacks/
│   │   ├── stack1/
│   │   │   ├── 1.png
│   │   │   ├── 2.png
│   │   │   └── ...
│   │   └── stack2/
│   │       ├── 1.png
│   │       ├── 2.png
│   │       └── ...
│   └── depth_maps/
│       ├── stack1.png
│       └── stack2.png
├── val_dataset2/
├── val_dataset3/
├── val_dataset4/
└── val_dataset5/
```

Key directory structure requirements:
- Each dataset has two main subdirectories: `image_stacks` and `depth_maps`
- In `image_stacks`, each scene has its own folder containing sequentially numbered images (e.g., 1.png, 2.png, ...)
- In `depth_maps`, each scene has a corresponding depth map with the same name as its stack folder (e.g., stack1.png for stack1 folder)
- All training and validation datasets follow the same structure as shown in the examples above
- Images should be in PNG, JPG, or BMP format
- Depth maps should be in grayscale PNG format

The framework supports up to 5 training datasets and 5 validation datasets simultaneously. You can control which datasets to use during training with the following flags:
- `--use_train_dataset_1` to `--use_train_dataset_5`
- `--use_val_dataset_1` to `--use_val_dataset_5`

During training, the framework will:
1. Train on all enabled training datasets
2. Validate on all enabled validation datasets separately
3. Save validation metrics for each dataset independently
4. Generate visualization results for each validation dataset

Training command example with multiple datasets:
```bash
python train.py \
    --train_stack "train_dataset1/image_stacks" \
    --train_depth_continuous "train_dataset1/depth_maps" \
    --train_stack_2 "train_dataset2/image_stacks" \
    --train_depth_continuous_2 "train_dataset2/depth_maps" \
    --train_stack_3 "train_dataset3/image_stacks" \
    --train_depth_continuous_3 "train_dataset3/depth_maps" \
    --train_stack_4 "train_dataset4/image_stacks" \
    --train_depth_continuous_4 "train_dataset4/depth_maps" \
    --train_stack_5 "train_dataset5/image_stacks" \
    --train_depth_continuous_5 "train_dataset5/depth_maps" \
    --val_stack "val_dataset1/image_stacks" \
    --val_depth_continuous "val_dataset1/depth_maps" \
    --val_stack_2 "val_dataset2/image_stacks" \
    --val_depth_continuous_2 "val_dataset2/depth_maps" \
    --val_stack_3 "val_dataset3/image_stacks" \
    --val_depth_continuous_3 "val_dataset3/depth_maps" \
    --val_stack_4 "val_dataset4/image_stacks" \
    --val_depth_continuous_4 "val_dataset4/depth_maps" \
    --val_stack_5 "val_dataset5/image_stacks" \
    --val_depth_continuous_5 "val_dataset5/depth_maps" \
    --batch_size 12 \
    --num_epochs 50 \
    --lr 1e-3 \
    --training_image_size 384
```

For detailed parameter descriptions, please refer to the source code.


## 📚 Citation

If you use this project in your research, please cite our papers:

```bibtex
@article{xie2025stackmff,
  title={StackMFF: end-to-end multi-focus image stack fusion network},
  author={Xie, Xinzhe and Qingyan, Jiang and Chen, Dong and Guo, Buyu and Li, Peiliang and Zhou, Sangjun},
  journal={Applied Intelligence},
  volume={55},
  number={6},
  pages={503},
  year={2025},
  publisher={Springer}
}

@article{xie2025multi,
  title={Multi-focus image fusion with visual state space model and dual adversarial learning},
  author={Xie, Xinzhe and Guo, Buyu and Li, Peiliang and He, Shuangyan and Zhou, Sangjun},
  journal={Computers and Electrical Engineering},
  volume={123},
  pages={110238},
  year={2025},
  publisher={Elsevier}
}

@article{xie2024swinmff,
  title={SwinMFF: toward high-fidelity end-to-end multi-focus image fusion via swin transformer-based network},
  author={Xie, Xinzhe and Guo, Buyu and Li, Peiliang and He, Shuangyan and Zhou, Sangjun},
  journal={The Visual Computer},
  pages={1--24},
  year={2024},
  publisher={Springer}
}
@article{xie2025lightmff,
  title={LightMFF: A Simple and Efficient Ultra-Lightweight Multi-Focus Image Fusion Network},
  author={Xie, Xinzhe and Lin, Zijian and Guo, Buyu and He, Shuangyan and Gu, Yanzhen and Bai, Yefei and Li, Peiliang},
  journal={Applied Sciences},
  volume={15},
  number={13},
  pages={7500},
  year={2025},
  publisher={MDPI}
}
@inproceedings{xie2024underwater,
  title={Underwater Three-Dimensional Microscope for Marine Benthic Organism Monitoring},
  author={Xie, Xinzhe and Guo, Buyu and Li, Peiliang and Jiang, Qingyan},
  booktitle={OCEANS 2024-Singapore},
  pages={1--4},
  year={2024},
  organization={IEEE}
}
```

## 🙏 Acknowledgments

We sincerely thank all the reviewers who have been responsible and have improved the quality of this study!
---

⭐ If you find this project helpful, please give it a star!
</div>

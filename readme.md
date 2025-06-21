# Incorporating Rather Than Eliminating: Achieving Fairness for Skin Disease Diagnosis Through Group-Specific Experts

This repository contains the implementation of a Mixture of Experts approach for addressing fairness in skin lesion classification tasks.

## Overview

This project implements a fairness-aware Mixture of Experts model that addresses bias in skin lesion classification across different demographic groups. The model uses multiple expert networks with a router mechanism that considers sensitive attributes during training to ensure fair performance across different skin types and demographic groups.

## Datasets

### Fitzpatrick17k
- **Description**: Large-scale dataset of skin lesions with Fitzpatrick skin type annotations
- **Classes**: 3, 9, or 114 classes (configurable)
- **Sensitive Attribute**: Skin color (light/dark binary classification)
- **Paper**: [Fitzpatrick17k: A Dataset for Evaluating Skin Lesion Classification](https://arxiv.org/abs/2008.07396)

### ISIC2019
- **Description**: International Skin Imaging Collaboration 2019 challenge dataset
- **Classes**: 9 classes
- **Sensitive Attributes**: Gender or age (configurable)
- **Paper**: [ISIC 2019 Challenge](https://challenge2019.isic-archive.com/)

## Usage

### Pre-training Standard Model
```bash
python pretrain.py \
    --dataset fitzpatrick17k \
    --image_dir /path/to/dataset_images/ \
    --csv_file_name /path/to/dataset.csv \
    --root /path/to/dataset/ \
    --epochs 200 \
    --lr 0.0001
```

### Training MoE Model

#### Fitzpatrick17k Dataset
```bash
python train_moe.py \
    --dataset fitzpatrick17k \
    --image_dir /path/to/fitzpatrick17k/dataset_images/ \
    --csv_file_name /path/to/fitzpatrick17k/fitzpatrick17k.csv \
    --root /path/to/fitzpatrick17k/ \
    --epochs 200 \
    --num_classes 114 \
    --lr 0.0001 \
    --n_experts 2 \
    --mi_weight 0.01
```

#### ISIC2019 Dataset
```bash
python train_moe.py \
    --dataset ISIC2019 \
    --image_dir /path/to/ISIC2019/ISIC_2019_Training_Input/ \
    --csv_file_name /path/to/ISIC2019/ISIC2019.csv \
    --root /path/to/ISIC2019/ \
    --epochs 200 \
    --fair_attr age \
    --lr 0.0001 \
    --n_experts 2
```

## File Structure

```
├── train_moe.py          # Main training script for MoE model
├── pretrain.py           # Pre-training script for standard model
├── model.py              # MoE model architecture definitions
├── fairness_metrics.py   # Fairness evaluation metrics
├── utils.py              # Utility functions
├── data_loader/          # Dataset loading modules
│   ├── fitzpatrick17k_data.py
│   ├── ISIC2019_data.py
│   └── data_utils.py
```

<!-- filepath: c:\VSCode\RT-DETR-Temporal\rtdetrv2_pytorch\README_L.md -->
# Training Configuration Guide

The training process is controlled by a main YAML configuration file that you pass to `tools/train.py` using the `-c` flag. This main file uses an `__include__` directive to inherit and combine settings from several other specialized configuration files. This makes the setup modular and easier to manage.

## Main Configuration File

This is the primary file for a specific training run. It brings together all the other necessary configuration pieces.

*   **Example:** `configs/rtdetr/rtdetr_r34vd_6x_uav.yml`
*   **Purpose:** Defines the output directory, specifies the model architecture details, and includes all other required configs.

```yaml
# Example from rtdetr_r34vd_6x_uav.yml
__include__: [
  '../dataset/uav_detection.yml',
  '../runtime.yml',
  './include/optimizer.yml',
  './include/rtdetr_r50vd.yml',
]

output_dir: ./output/rtdetr_r34vd_6x_coco

PResNet:
  depth: 34
# ... more model-specific settings
```

---

## Configuration Breakdown

Here is a breakdown of the key configuration files and the parameters they control:

### 1. Training Schedule & Optimizer

This file controls how the model is trained, including the number of epochs, learning rate, and optimization algorithm.

*   **File:** `configs/rtdetr/include/optimizer.yml`
*   **Key Settings:**
    *   `epoches`: The total number of training epochs.
    *   `optimizer`: The optimization algorithm (e.g., `AdamW`), its learning rate (`lr`), and `weight_decay`.
    *   `lr_scheduler`: The learning rate scheduling policy (e.g., `MultiStepLR`).

### 2. Dataset Configuration

This file points to the training and validation data and defines dataset-specific parameters.

*   **File:** `configs/dataset/uav_detection.yml`
*   **Key Settings:**
    *   `num_classes`: The number of object classes in your dataset.
    *   `train_image_dir`, `val_image_dir`: Paths to the training and validation image folders.
    *   `train_anno_path`, `val_anno_path`: Paths to the training and validation annotation files (e.g., COCO format JSON).

### 3. Data Loading & Augmentation

These settings, found within the dataset configuration file, define how data is loaded, batched, and transformed.

*   **File:** `configs/dataset/uav_detection.yml`
*   **Key Settings (under `train_dataloader` and `val_dataloader`):**
    *   `batch_size`: Number of samples processed in each batch.
    *   `num_workers`: Number of CPU threads to use for data loading.
    *   `transforms`: A sequence of data augmentation and preprocessing steps applied to each image. These correspond to classes defined in `src/data/transforms.py`.

### 4. Model Architecture

The base architecture of the model is defined in a dedicated file, which can be overridden by the main configuration file.

*   **Base Model File:** `configs/rtdetr/include/rtdetr_r50vd.yml`
*   **Main Config Override:** `configs/rtdetr/rtdetr_r34vd_6x_uav.yml`
*   **Key Settings:**
    *   `PResNet.depth`: Defines the depth of the ResNet backbone (e.g., 18, 34, 50).
    *   `HybridEncoder`: Configuration for the encoder part of the transformer.
    *   `RTDETRTransformer`: Configuration for the transformer decoder.

### 5. Runtime Settings

This file contains general runtime configurations for the training environment.

*   **File:** `configs/runtime.yml`
*   **Key Settings:**
    *   `use_amp`: (Boolean) Whether to use Automatic Mixed Precision for faster training.
    *   `use_ema`: (Boolean) Whether to use Exponential Moving Average for model weights.
    *   `sync_bn`: (Boolean) Whether to synchronize batch normalization across multiple GPUs.

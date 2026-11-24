# Skin Lesion Classification on HAM10000: Custom CNN vs ImageNet-Pretrained Models

A comparative study of deep-learning architectures for seven-class dermoscopic skin-lesion classification using the HAM10000 dataset.

## Research Objective

**Research question:** How do a custom CNN trained from scratch and ImageNet-pretrained architectures compare in seven-class dermoscopic skin-lesion classification under a common evaluation protocol?

We evaluate four architectures:

| Model | Training approach |
|---|---|
| Custom CNN (~2.4M parameters) | Trained from scratch |
| ResNet50 | ImageNet-pretrained, classifier-head training followed by fine-tuning |
| DenseNet201 | ImageNet-pretrained, classifier-head training followed by fine-tuning |
| EfficientNet-B3 | ImageNet-pretrained, classifier-head training followed by fine-tuning |

The study focuses on class-imbalanced classification performance, particularly macro-F1 and melanoma recall, rather than relying on accuracy alone.

**Important scope:** This experiment compares different architectures and training approaches. Because the scratch-trained CNN and pretrained models have different architectures, the comparison does not independently isolate the causal effect of ImageNet pretraining.

## Dataset

We use the HAM10000 (Human Against Machine with 10000 training images) dataset, containing 10,015 dermoscopic images across seven diagnostic categories.

| Class | Description |
|---|---|
| akiec | Actinic keratoses and intraepithelial carcinoma |
| bcc | Basal cell carcinoma |
| bkl | Benign keratosis-like lesions |
| df | Dermatofibroma |
| mel | Melanoma |
| nv | Melanocytic nevi |
| vasc | Vascular lesions |

### Dataset split

The project uses a fixed lesion-level split shared across all models.

| Split | Images | Percentage |
|---|---:|---:|
| Training | 6,376 | 63.66% |
| Validation | 1,613 | 16.11% |
| Testing | 2,026 | 20.23% |
| **Total** | **10,015** | **100%** |

Images belonging to the same lesion are assigned to the same split. The supplied CSV files have zero overlapping `lesion_id` values across training, validation and testing.

This helps prevent lesion-level data leakage and provides a more reliable evaluation than randomly splitting individual images.

The split files are located in `splits/`.

## Experimental Protocol

The following settings are shared across the experiments:

- Input resolution: 224 × 224 pixels
- Effective batch size: 16
- Optimizer: AdamW
- Learning-rate scheduler: ReduceLROnPlateau
- Weight decay: 1e-4
- Loss: Class-weighted cross-entropy by default
- Checkpoint selection: Validation loss
- Random seed: 42 by default

The transfer-learning models use two training stages:

1. **Classifier-head training:** Freeze the pretrained backbone and train the newly added classification head.
2. **Fine-tuning:** Unfreeze the backbone and train the complete model.

The custom CNN is trained from scratch.

### Training configuration

| Model | Head epochs | Fine-tuning epochs | Maximum total |
|---|---:|---:|---:|
| Custom CNN | 0 | 80 | 80 |
| ResNet50 | 5 | 10 | 15 |
| DenseNet201 | 5 | 10 | 15 |
| EfficientNet-B3 | 5 | 10 | 15 |

These are maximum epoch budgets. Early stopping may terminate training earlier.

Model-specific learning rates and patience values are defined in `config.py`. The micro-batch size can vary by architecture, while gradient accumulation maintains an effective batch size of 16.

**Reproducibility note:** The original ResNet50 Colab experiment has some unrecovered hyperparameters. Its historical result should not be treated as a strictly controlled comparison with the other models until ResNet50 is retrained using the current configuration.

## Evaluation Metrics

Because HAM10000 is class-imbalanced, accuracy alone is insufficient.

The primary evaluation metric is **macro-F1**, which gives equal importance to each diagnostic class.

Additional metrics include:

- Overall accuracy
- Macro precision, recall and F1-score
- Per-class precision, recall and F1-score
- Melanoma recall
- Multiclass ROC-AUC, where computable
- Confusion matrices

The test split is held out from model fitting, checkpoint selection and hyperparameter tuning. It is evaluated after model selection for final performance reporting.

## Explainability and Statistical Analysis

The project includes additional analysis tools:

### Grad-CAM

Grad-CAM visualisations are generated to examine model attention in selected cases:

- Correctly classified examples
- Missed melanomas
- False melanoma predictions

These visualisations support qualitative error analysis. They should not be interpreted as proof of clinical reasoning or model reliability.

### McNemar's Test

Pairwise McNemar testing is used to compare the prediction errors of two models on the same test examples.

This provides a statistical comparison of their classification disagreements.

## Repository Structure

```text
skin-lesion-classification/
│
├── common.py
├── config.py
├── train.py
├── run_all.py
├── gradcam_analysis.py
├── mcnemar_test.py
│
├── metrics_utils.py
├── compare_results.py
├── make_random_split.py
│
├── splits/
│   ├── train_split.csv
│   ├── validation_split.csv
│   └── test_split.csv
│
├── data/
│   └── HAM10000 image folders
│
├── outputs/
│   ├── checkpoints/
│   ├── reports/
│   ├── plots/
│   └── gradcam/
│
├── requirements.txt
├── .gitignore
└── README.md
```

Dataset images, cached images and large model checkpoints should not be committed to GitHub.

## Requirements

- Windows or Linux
- Python 3.10 or 3.11
- PyTorch and torchvision
- NVIDIA GPU recommended for practical training
- Approximately 6 GB of free disk space, depending on the dataset and image cache

The code can fall back to CPU execution if CUDA is unavailable, although training will be considerably slower.

## Installation

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd skin-lesion-classification
```

### 2. Create a virtual environment

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install PyTorch

Install the PyTorch build appropriate for your operating system, NVIDIA driver and CUDA compatibility.

For example, a CUDA 12.1 installation:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 4. Install project dependencies

```bash
pip install -r requirements.txt
```

### 5. Download the dataset

Download HAM10000 from:

- Kaggle: https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000
- Harvard Dataverse: https://doi.org/10.7910/DVN/DBW86T

Extract the image folders under `data/`.

For example:

```text
data/
├── HAM10000_images_part_1/
└── HAM10000_images_part_2/
```

The project uses the supplied split CSV files, so the original metadata CSV is not required for the training pipeline if the split files are present and complete.

Do not upload the original dataset images to this repository. Check the dataset terms before using or redistributing the data.

### 6. Verify the installation

Run a short training smoke test:

```bash
python train.py --model custom_cnn --smoke-test
```

This is a setup check, not a valid experiment or reported result.

## Running the Experiments

### Run the default study

```bash
python run_all.py
```

The default configuration runs:

- Custom CNN
- DenseNet201
- EfficientNet-B3

It uses weighted cross-entropy and seed 42.

### Run an individual model

```bash
python train.py --model custom_cnn
python train.py --model densenet201
python train.py --model efficientnet_b3
```

### Include ResNet50

```bash
python run_all.py --include-resnet50
```

This retrains ResNet50 using the current local configuration.

### Compare loss functions

```bash
python run_all.py --losses weighted_ce focal
```

This compares class-weighted cross-entropy with focal loss.

### Repeat experiments with multiple seeds

```bash
python run_all.py --seeds 42 43 44
```

Multiple runs can be used to report mean and standard deviation across seeds.

### Optional image-level split ablation

```bash
python run_all.py --ablation resnet50
```

This is an optional leakage experiment using a separate random image-level split. It is not part of the primary lesion-level evaluation and should be clearly labelled as an ablation.

## Outputs

Training produces model-specific artifacts under `outputs/`, including:

- Best model checkpoints
- Training history
- Classification reports
- Confusion matrices
- Per-image predictions
- Training curves
- Grad-CAM visualisations

The comparison script aggregates results across available runs.

Generated artifacts should be inspected before including them in a paper. Only report results that were actually produced by completed experiments.

## Current Experimental Status

The original ResNet50 Colab run produced the following historical result:

| Metric | ResNet50 |
|---|---:|
| Accuracy | 0.804 |
| Macro-F1 | 0.678 |
| ROC-AUC | 0.954 |
| Melanoma recall | 0.625 |

These values were imported from the original run's confusion matrix and summary JSON. The original trained weights and per-image predictions are not currently included.

The Custom CNN, DenseNet201 and EfficientNet-B3 experiments are pending execution under the current local pipeline.

**The historical ResNet50 result is provided as a reference, not as a directly controlled comparison.** Retraining ResNet50 with the current configuration is necessary for a more consistent comparison.

## Limitations

- The study uses a single dermoscopic image dataset.
- There is no independent external validation dataset.
- The experiment does not establish clinical diagnostic performance.
- Rare classes have relatively few test examples. For example, dermatofibroma (`df`) has 22 test images and vascular lesions (`vasc`) have 30.
- Grad-CAM is a qualitative interpretability method, not proof of model reasoning.
- GPU hardware, mixed precision and backend operations may introduce small differences between runs.
- The historical ResNet50 configuration is not fully recoverable.

The results should be interpreted as a comparative machine-learning experiment, not as a clinically validated diagnostic system.

## Dataset Citation

Please cite the original HAM10000 publication when using the dataset:

Tschandl, P., Rosendahl, C. & Kittler, H. (2018). The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions. *Scientific Data*, 5, 180161.

https://doi.org/10.1038/sdata.2018.161

The dataset is distributed under non-commercial terms. Review its licence and terms of use before using it.

## Disclaimer

This project is intended for academic and research purposes only. It is not a medical device and must not be used for clinical diagnosis or treatment decisions.

## License

The code in this repository is released under the MIT License (see `LICENSE`).
The HAM10000 dataset is not included and has its own license (check the dataset
page on Harvard Dataverse before use). Please cite:

Tschandl, P., Rosendahl, C., Kittler, H. The HAM10000 dataset, a large
collection of multi-source dermatoscopic images of common pigmented skin
lesions. Sci. Data 5, 180161 (2018).
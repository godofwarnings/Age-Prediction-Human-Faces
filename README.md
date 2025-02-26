# Age Prediction from Facial Images

## Overview

The goal is to predict the age of a person given an image of their face. This task has various real-world applications, including:

- Age-restricted content filtering
- Personalized marketing
- Security systems with age verification
- Medical diagnostics

## Project Structure

```
Age-Prediction-on-Human-Faces/
│
├── README.md
├── predict.py
├── requirements.txt
├── SMAI Assignment 4.pdf    
│
├── results/
│   └── BestResults-ConvNeXT_Small.csv
│
├── src/
│   ├── dataset.py
│   ├── model.py
│   └── train.py
│
└── data/
    ├── train/
    ├── test/
    ├── train.csv
    └── submission.csv
```

## Setup and Installation

1. Clone this repository
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Data

The dataset is structured as follows:

- `train/`: Directory containing training images
- `test/`: Directory containing test images
- `train.csv`: CSV file with training data annotations
- `submission.csv`: CSV file for submission format

## Model

Our approach leverages transfer learning using various architectures:

- CNN-based models
- Vision Transformer (ViT) based architectures
- All models are pretrained on ImageNet
- Best performing model: ConvNeXt Small

## Training

To train the model, execute:

```bash
python src/train.py
```

The training pipeline:

1. Loads and preprocesses the data
2. Initializes the selected model architecture
3. Trains using Adam optimizer and L1 Loss
4. Saves the best model based on validation performance
5. Logs results to Weights & Biases (wandb)

## Prediction

Generate predictions on the test set by running:

```bash
python predict.py
```

This will create a CSV file containing age predictions for all test images.

## Results

This implementation achieved significant success:

- Model: ConvNeXt Small
- Ranking: 7th out of 200 participants in the Kaggle competition
- Detailed results available in: `results/BestResults-ConvNeXT_Small.csv`

## Acknowledgements
- PyTorch library
- Pretrained models from torchvision
- Weights & Biases (wandb) for experiment tracking and visualization

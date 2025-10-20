# Homework 2 - Toan Ly

## Overview
This assignment implements multiple neural network models for MNIST digit classification using PyTorch. 

## Environment setup
This project is run under `Python 3.11`. You can install dependencies by running this command:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running the code
### Training
The training for all models is in `src/run_training.ipynb`. It's recommended to train on Google Colab for GPU utilization. After running the script, `all_files.zip` will be saved and you will need to run:
```bash
make unzip
```
The script will organize all needed files for evaluation, including the train/val loss, accuracy, f1, and model weights

### Evaluation
```bash
python src/evaluation.py
```
This will load all trained models, run on the test set, and plot the results

## Folder structure
- `results/`: stores loss curves and roc curves
- `weights/`: stores all models' weights, or the best model's weights
- `src/`: main folder containing all the codes
- `docs/`: stores the report PDF for submission

In `src/evaluation.py`, `test_model()` function loads the trained model and evaluate on the test set

### Notebooks:
- `src/run_eda.ipynb`: notebook for Exploratory Data Analysis
- `src/run_training.ipynb`: notebook for training and saving results

## Model Details
- **DNN** (`src/models/dnn.py`): 2 hidden layers (128 and 64)
- **CNN** (`src/models/cnn.py`): 2 convolution layers (16 and 32 filters) and a fully connected layer (64)
- **ResNet18** (`src/models/resnet.py`): 4 residual blocks (2-2-2-2) ending with global average pooling and a fully connected layer (512)
- **VGG16** (`src/models/vgg.py`): Stacked 3x3 convolutions and fully connected layers

## Model Comparison

| Model             | $F1$  | $AUC$ 
|-------------------|-------|-------
| DNN               | 0.980 | 0.9996
| CNN               | 0.991 | 0.9999
| ResNet18          | 0.993 | 1.0
| VGG16             | 0.995 | 0.9999

**Best Performance**:
##### `VGG16`:
- Accuracy: 0.995
- F1: 0.995
- AUC: 0.9999

## Inference
Below are the example inference of each model on the test set:
![DNN predictions](/homework2/results/demo/DNN.png)
![CNN predictions](/homework2/results/demo/CNN.png)
![ResNet predictions](/homework2/results/demo/ResNet.png)
![VGG predictions](/homework2/results/demo/VGG.png)
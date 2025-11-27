# Homework 4 - Toan Ly

## Overview
This assignment performs graph-level classification on the **ENZYMES** dataset using GCN (Graph Convolutional Network)

## Environmental setup

This project is run under `Python 3.11`. You can install dependencies by running this command:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running the code
### Training and Evaluation
The training and evaluation for all models is in `src/train.py`. All training was done locally. The code will train all models with training and validation sets, and evaluate on the test set, plot and save results at `results/`

### Folder structure
- `docs/`: report PDF for submission
- `src/`: 
    - `eda.ipynb`: jupyter notebook for EDA
    - `preprocess.py`: code to preprocess the ENZYMES dataset
    - `utils.py`: utility functions
    - `train.py`: main train and evaluate code
- `weights/`: all models' weights, or the best model's weights 

Additionally, in `src/train.py`, `test_model()` function loads the trained model and evaluate on the test set

## Model Comparison

| Model             |Parameters| F1      |  Accuracy | AUC
|-------------------|----------|---------|-----------|-----
| GCN-1 layer       |205K      | 0.585   | 0.600     | 0.851
| GCN-2 layer       |271K      |**0.833**| **0.833** | **0.912**
| GCN-3 layer       |337K      |  0.707  | 0.717     | 0.895

### Best Performance
`GCN-2 layer`:
- F1: 0.833
- Accuracy: 0.833

![Loss](/homework4/results/gcn[2]_loss.png)
![ROC-Curve](/homework4/results/gcn[2]_roc.png)
# Homework 1 - Toan Ly

## Overview
This assignment trains and evaluates a linear regression and several fully connected DNN architectures to predict **cancer motality rates** (`TARGET_deathRate`)

## Dataset
The dataset is in `data/` directory
```
data
├── cancer_reg-1.csv
└── processed
    ├── X_test.csv
    ├── X_train.csv
    ├── X_val.csv
    ├── y_test.csv
    ├── y_train.csv
    └── y_val.csv
```

## Environment setup
This project is run under `Python 3.11`. You can install dependencies by running this command:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running the code
### Run all models with specific learning rate
```bash
make run LR=0.01
```

### Run all models with all defined learning rates (0.1, 0.01, 0.001, 0.0001)
```bash
make run-all
```

### Clean generated files
```bash
make clean
```

## Folder structures
- `data/`: dataset folder
- `figures/`: folder to store loss curves, performance results, and training screenshots with timestamp
- `weights/`: folder to store linear regression and best dnn weights
- `src/`: main folder to store codes (model, train, and utilities) and jupyter notebooks for EDA, preprocessing and evaluation

Additionally, `src/utils/evaluation.py` includes `test_model()` to load model and evaluate on the test set

### Notebooks:
- `src/run_eda.ipynb`: notebook for Exploratory Data Analysis
- `src/run_preprocessing.ipynb`: notebook for data preprocessing, this is where I generated `src/data/processed`
- `src/run_linear_regression.ipynb`: notebook for training and evaluating linear regression with 3 versions
- `src/run_dnn.ipynb`: notebook for training and evaluating all dnn models

`src/main.py` is the main python script for this assignment

## Model Details
### Linear Regression (`src/models/linear_regression.py`)
Although the python file contains 3 linear regression versions (sklearn, numpy, and PyTorch), I realized that numpy and torch versions required careful tuning for epochs and learning rates to be able to match sklearn performance. Therefore, I decided to use sklearn for further training and evaluation

- **Performance**:
    - MAE: 10.40
    - MSE: 188.01
    - $R^2$: 0.78

### DNN (`src/models/dnn)
DNN is implemented using `PyTorch`. It supports:
- Configurable hidden layer sizes (e.g., `[16]`, `[30, 8]`, etc.)
- Activation functions (`ReLU`, `LeakyReLU`, `Sigmoid`, `Tanh`)
- Weight initilization 
- SGD optimizer with momentum
- Batch Normalization and Dropout (optional)
- L2 regularization, weight decay (optional)
- Gradient clipping
- Early stopping

**Architectures Tested**:
- DNN-16
- DNN-30-8
- DNN-30-16-8
- DNN-30-16-8-4
- DNN-64-30-16-8 (custom)
- DNN-128-64-32

## Model Comparison

| Model             | $R^2$|
|-------------------|------|
| Linear Regression | 0.78 |
| DNN-16            | 0.81 |
| DNN-30-8          | 0.88 |
| DNN-30-16-8       | 0.81 |
| DNN-30-16-8-4     | 0.79 |
| DNN-64-30-16-8    | 0.77 |
| DNN-128-64-32     | 0.74 |

**Best Performance**:

##### `DNN-30-8`:
- MAE: 7.36
- MSE: 101.11
- $R^2$: 0.88
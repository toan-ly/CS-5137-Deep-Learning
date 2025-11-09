# Homework 3 - Toan Ly

## Overview
This assignment implements UNet and ResUNet for Retinal Vessel Segmentation using PyTorch.

## Environment setup
This project is run under `Python 3.11`. You can install dependencies by running this command:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running the code
### Training
The training for all models is in `src/notebooks/run_training.ipynb`. It's recommended to train on Google Colab for GPU utilization. After running the script, `checkpoints.zip` will be downloaded and you will need to run:
```bash
make unzip DIR=your_choice
```
The script will unzip and organize all needed files for evaluation, including the train/val loss, dice, iou, and model weights

### Evaluation
```bash
python src/main.py
```
This will load all trained models, run on the test set, plot and save the results at `figures/eval/`

Additionally, running `src/notebooks/run_evaluation.ipynb` will plot and save the comparisons between predictions of all models

### Folder structure
- `figures/`:
    - `augmentation_previews/`: sample image after each augmentation 
    - `eval/`: predictions for each model on the test set
    - `comparisons/`: predictions for each model side by side for comparison
    - `screenshows/`: screenshots of iterations of model training and testing with current timestamp

- `configs/`: YAML training configs
- `checkpoints/`: all models' weights, or the best model's weights
- `docs/`: report PDF for submission
- `src/`:
    - `data/`: codes for data augmentations
    - `models/`: implementations of model architecture
    - `tests/`: codes for testing augmentations
    - `notebooks/`: jupyter notebooks for EDA, training, and evaluation
    - `utils/`: utility functions
    - `trainer.py` and `train.py`: helper functions for training models
    - `main.py`: main code for testing all models

Additionally, in `src/main.py`, `test()` function loads the trained model and evaluate on the test set

## Model Comparison
| Model             |Parameters| $Dice$  | $IoU$ 
|-------------------|----------|---------|-------
| UNet 2 blocks     |1.87M     | 0.791   | 0.655
| UNet 3 blocks     |7.19M     |**0.825**| **0.703**
| UNet 4 blocks     |31.04M    | 0.823   | 0.699
| ResUNet 3 blocks  |8.05M     | 0.820   | 0.695
| ResUNet 4 blocks  |32.45M    | 0.823   | 0.700

**Best Performance**:
##### `UNet 3 blocks`:
- Dice: 0.825
- IoU: 0.703

![Loss](/homework3/figures/eval/UNET[64-3]_norm_nearest/train_curve.png)

## Demo
#### Predictions of best model:
![1](/homework3/figures/eval/UNET[64-3]_norm_nearest/1.png)
![6](/homework3/figures/eval/UNET[64-3]_norm_nearest/6.png)

#### Comparisons between all models:
![comparison](/homework3/figures/comparisons/11.png)
![comparison](/homework3/figures/comparisons/9.png)
![comparison](/homework3/figures/comparisons/13.png)
import matplotlib.pyplot as plt
import numpy as np
import torch
import pandas as pd
from pathlib import Path
from utils.plots import *



if __name__ == "__main__":
    ROOT = Path(__file__).parent.parent
    DATA_DIR = ROOT / 'data'
    RESULTS_DIR = ROOT / 'results'

    history = pd.read_csv(RESULTS_DIR / 'histories.csv')

    plot_loss_by_lr(history, save_dir=RESULTS_DIR)





import matplotlib.pyplot as plt
import numpy as np
import torch
import pandas as pd
from pathlib import Path
from utils import *


if __name__ == "__main__":
    ROOT = Path(__file__).parent.parent
    DATA_DIR = ROOT / 'data'
    RESULTS_DIR = ROOT / 'results'

    loss_df = RESULTS_DIR / 'histories.csv'
    history = pd.read_csv(loss_df)

    plot_loss_by_lr(history, save_dir=RESULTS_DIR)





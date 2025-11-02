import os
import glob
import math
import random
import torch
from pathlib import Path

from .data.dataset import *
from .utils import plot_loss

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / 'data'
CKPT_DIR = ROOT / 'checkpoints'
RESULTS_DIR = ROOT / 'figures' / 'eval'
METRIC_PATH = RESULTS_DIR / 'metrics.csv'

os.makedirs(RESULTS_DIR, exist_ok=True)


USE_GREEN = True
BATCH_SIZE = 4
IM_SIZE = 512

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model(model, weight_path):
    model.load_state_dict(torch.load(weight_path))
    model.eval()
    return model

def main():
    test_loader = make_test_loader(
        data_root=DATA_DIR,
        im_size=IM_SIZE,
        batch_size=BATCH_SIZE,
        use_green_channel=USE_GREEN
    )
    print(f"Number of test samples: {len(test_loader.dataset)}\n")

    ckpt_dirs = glob.glob(str(CKPT_DIR / '*' / 'best_model.pth'))
    print(ckpt_dirs)




if __name__ == "__main__":
    main()

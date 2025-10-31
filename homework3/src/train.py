
import os
import argparse

from src.data import make_loaders, make_test_loader
from src.models import UNet
from src.utils import *
from src.trainer import Trainer

import pandas as pd

def parse_args():
    ap = argparse.ArgumentParser("Training script")

    # Data
    ap.add_argument("--data_root", required=True, help="dataset_root with train/ and test/")
    ap.add_argument("--img_size", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--cache_rate", type=float, default=0.0, help="MONAI CacheDataset rate (0..1)")
    ap.add_argument("--use_green_channel", action='store_true', help="Use only green channel of the input images")
    ap.add_argument("--val_ratio", type=float, default=0.3, help="Validation data ratio")

    # Model
    ap.add_argument("--in_channels", type=int, default=3)
    ap.add_argument("--out_channels", type=int, default=1)
    ap.add_argument("--base_channels", type=int, default=32)
    ap.add_argument("--depth", type=int, choices=[2,3], default=2)
    ap.add_argument("--dropout", type=float, default=0.0)

    # Optimization
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--loss", choices=["bce","dice","bce_dice"], default="bce_dice")
    ap.add_argument("--scheduler", choices=["step","plateau","none"], default="step")
    ap.add_argument("--optimizer", choices=["adam","adamw","sgd"], default="adamw")

    # Trainer features
    ap.add_argument("--early_stopping", action='store_true')
    ap.add_argument("--early_stopping_patience", type=int, default=10)
    ap.add_argument("--activation", choices=["relu","leaky_relu","elu"], default="relu")
    ap.add_argument("--up_mode", choices=["transpose","bilinear"], default="bilinear")

    # I/O
    ap.add_argument("--save_dir", default="checkpoints")
    ap.add_argument("--save_plots_path", default="results/predictions")
    ap.add_argument("--device", default=None, help="force device, e.g., 'cuda'|'mps'|'cpu'")
    ap.add_argument("--metrics_csv", default=None, help="optional CSV to append metrics per epoch")
    return ap.parse_args()

def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.save_plots_path, exist_ok=True)

    train_loader, val_loader = make_loaders(
        data_root=args.data_root,
        im_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        cache_rate=args.cache_rate,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    features = [args.base_channels * (2 ** i) for i in range(args.depth)]
    model = UNet(
        n_channels=args.in_channels,
        n_classes=args.out_channels,
        features=features,
        activation=args.activation,
        dropout=args.dropout,
        up_mode=args.up_mode,
    )

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=None,
        device=device,
        loss=args.loss,
        lr=args.lr,
        early_stopping=args.early_stopping,
        early_stopping_patience=args.early_stopping_patience,
        scheduler='step',
    )

    history = trainer.fit(
        epochs=args.epochs,
        save_model_path=os.path.join(args.save_dir, "best_model.pth"),
        save_plots_path=args.save_plots_path,
        verbose=True,
    )

    history_df = pd.DataFrame(history)
    history_df['epoch'] = history_df.index + 1
    history_df.to_csv(os.path.join(args.save_dir, "training_history.csv"))


if __name__=='__main__':
    main()

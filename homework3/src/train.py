
import os
import argparse
import yaml

from src.data import make_loaders, make_test_loader
from src.models import UNet
from src.utils import *
from src.trainer import Trainer

import pandas as pd

def build_parsers():
    p = argparse.ArgumentParser("Training script", fromfile_prefix_chars='@')
    p.add_argument("--config", type=str, default=None)
    # Data
    p.add_argument("--data_root", required=True)
    p.add_argument("--img_size", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--cache_rate", type=float, default=0.0)
    p.add_argument("--use_green_channel", action='store_true')
    p.add_argument("--val_ratio", type=float, default=0.2)
    p.add_argument("--use_patch", action='store_true')

    # Model
    p.add_argument("--in_channels", type=int, default=3)
    p.add_argument("--out_channels", type=int, default=1)
    p.add_argument("--base_channels", type=int, default=32)
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--activation", default="relu")
    p.add_argument("--up_mode", choices=["transpose","bilinear"], default="bilinear")

    # Optimization
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--loss", default="dice")
    p.add_argument("--scheduler", default="step")
    p.add_argument("--optimizer", default="adamw")

    # Training
    p.add_argument("--early_stopping", action='store_true')
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--save_dir", default="checkpoints")
    p.add_argument("--save_plots_path", default="results/predictions")
    p.add_argument("--device", default=None)

    return p

def parse_args():
    p = build_parsers()

    # Get training configs from yaml file if provided
    args, _ = p.parse_known_args()
    if args.config:
        with open(args.config, "r") as f:
            cfg = yaml.safe_load(f) or {}
        valid = {a.dest for a in p._actions}
        unknown = [k for k in cfg.keys() if k not in valid]
        if unknown:
            raise ValueError(f"Unknown YAML config key(s): {unknown}")
        p.set_defaults(**cfg)
    args = p.parse_args()
    
    if args.use_green_channel:
        args.in_channels = 1

    return args

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
        use_green_channel=args.use_green_channel,
        use_patch=args.use_patch,
    )

    for batch in train_loader:
        print(f"Image batch shape: {batch['image'].shape}")
        print(f"Mask batch shape: {batch['mask'].shape}")
        break

    print('Starting training...')

    features = [args.base_channels * (2 ** i) for i in range(args.depth + 1)]
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
        patience=args.patience,
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

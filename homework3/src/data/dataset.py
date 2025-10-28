from pathlib import Path
import os, glob, random
from monai.data import Dataset, CacheDataset, DataLoader
from .aug import get_transforms

def collect_pairs(root):
    """
    Collect image-mask pairs from the given root directory.
    """
    img_dir = os.path.join(root, 'image')
    mask_dir = os.path.join(root, 'mask')
    images = sorted(glob.glob(os.path.join(img_dir, '*')))
    masks = sorted(glob.glob(os.path.join(mask_dir, '*')))

    assert len(images) == len(masks) and len(images) > 0, f'Issue with dataset in {root}'
    return [{'image': i, 'mask': m} for i, m in zip(images, masks)]

def split_data(pairs, val_ratio=0.2, seed=42):
    """
    Split data into training and validation sets.
    """
    random.seed(seed)
    random.shuffle(pairs)
    n_val = int(len(pairs) * val_ratio)
    val_data = pairs[:n_val]
    train_data = pairs[n_val:]
    return train_data, val_data

def make_loaders(
    data_root,
    im_size=512,
    normalize=True,
    batch_size=4,
    num_workers=2,
    cache_rate=0.0,
    val_ratio=0.2,
    seed=42,
):
    """
    Create training and validation data loaders.
    """
    train_root = os.path.join(data_root, 'train')
    train_data = collect_pairs(train_root)
    train_data, val_data = split_data(train_data, val_ratio, seed)

    train_transforms = get_transforms(im_size, normalize, is_train=True)
    val_transforms = get_transforms(im_size, normalize, is_train=False)

    if cache_rate > 0:
        train_ds = CacheDataset(data=train_data, transform=train_transforms, cache_rate=cache_rate, num_workers=num_workers)
        val_ds = CacheDataset(data=val_data, transform=val_transforms, cache_rate=cache_rate, num_workers=num_workers)
    else:
        train_ds = Dataset(data=train_data, transform=train_transforms)
        val_ds = Dataset(data=val_data, transform=val_transforms)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader

def make_test_loader(
    data_root,
    im_size=512,
    normalize=True,
    batch_size=1,
    num_workers=2,
):
    """
    Create test data loader.
    """
    test_root = os.path.join(data_root, 'test')
    test_data = collect_pairs(test_root)

    test_transforms = get_transforms(im_size, normalize, is_train=False)
    test_ds = Dataset(data=test_data, transform=test_transforms)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return test_loader

if __name__ == "__main__":
    data_root = Path(__file__).parent.parent.parent / 'data'
    train_loader, val_loader = make_loaders(data_root, im_size=512, normalize=True, batch_size=2, num_workers=0, cache_rate=0.5)
    test_loader = make_test_loader(data_root, im_size=512, normalize=True, batch_size=1, num_workers=0)
    for batch in train_loader:
        print(batch['image'].shape, batch['mask'].shape)
        break
    for batch in val_loader:
        print(batch['image'].shape, batch['mask'].shape)
        break
    for batch in test_loader:
        print(batch['image'].shape, batch['mask'].shape)
        break
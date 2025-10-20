import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

def load_test_data(data_dir, batch_size=512, use_cached=True):
    """
    Load MNIST test data

    Args:
        data_dir: directory where data is stored
        batch_size: batch size for DataLoader
        use_cached: if True, load pre-saved tensors; else, load from torchvision datasets
    """
    if use_cached:
        X = torch.load(f'{data_dir}/X_test.pt', map_location='cpu')
        y = torch.load(f'{data_dir}/y_test.pt', map_location='cpu')
        test_loader = DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=False)
    else:
        T = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        test_dataset = datasets.MNIST(root=data_dir, train=False, transform=T, download=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return test_loader

def load_model(model, model_path, device):
    """
    Load a model from a file and move it to the specified device.

    Args:
        model: the model instance to load the state dict into
        model_path: path to the saved model file
        device: cpu or cuda
    """
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model
import pandas as pd
import torch

def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        print(f"Warning: {e}. Skip for now, trying with 'latin1' encoding.")
        df = pd.read_csv(file_path, encoding='latin1')
        return df

def load_processed_data(file_path):
    X_train = pd.read_csv(f'{file_path}/X_train.csv').values
    y_train = pd.read_csv(f'{file_path}/y_train.csv').values
    X_val = pd.read_csv(f'{file_path}/X_val.csv').values
    y_val = pd.read_csv(f'{file_path}/y_val.csv').values
    X_test = pd.read_csv(f'{file_path}/X_test.csv').values
    y_test = pd.read_csv(f'{file_path}/y_test.csv').values

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    print(f'X_train shape: {X_train.shape}, y_train shape: {y_train.shape}')
    print(f'X_val shape: {X_val.shape}, y_val shape: {y_val.shape}')
    print(f'X_test shape: {X_test.shape}, y_test shape: {y_test.shape}')

    return X_train, X_val, X_test, y_train, y_val, y_test


import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import torch


def load_scaler(file_path):
    """
    Load saved scaler if used for target variable
    """
    scaler = joblib.load(file_path)
    return scaler
    
def test_model(model, X, y, is_scaled=False):
    """
    Evaluate model on test set and print MAE, MSE, R² scores

    Args:
        model: trained model (DNN or Linear Regression)
        X: test features, shape: (n_samples, n_features)
        y: true target values, shape: (n_samples, 1)
        is_scaled: whether the target values were scaled during training
    Returns:
        mae: Mean Absolute Error
        mse: Mean Squared Error
        r2: R² score
    """
    if is_scaled:
        scaler = load_scaler('../models/y_scaler.pkl')
    print("\n=========== Model Evaluation ===========")
    y_pred = model.predict(X)
    if is_scaled:
        y_pred = scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
        y = scaler.inverse_transform(y.reshape(-1, 1)).flatten()
    mae = mean_absolute_error(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)

    print(f'Mean Absolute Error (MAE): {mae:.4f}')
    print(f'Mean Squared Error (MSE): {mse:.4f}')
    print(f'R^2 Score: {r2:.4f}')
    print("="*40)

    return mae, mse, r2

def plot_loss(train_losses, val_losses, title='', save_path=None, log=False):
    """
    Plot training and validation loss curves

    Args:
        train_losses: list of training losses
        val_losses: list of validation losses
        title: title of the plot
        save_path: path to save the plot
        log: whether to use logarithmic scale for y-axis
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('MSE Loss')
    plt.title(title)
    plt.legend()
    plt.grid()
    if log:
        plt.yscale('log')
        # plt.title(title + ' (Log Scale)')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    # plt.close()

def save_linear_regression(model, path):
    """
    Save linear regression model
    """
    joblib.dump(model, path)
    # print(f'Linear regression model saved to {path}')
    return path

def save_dnn(model, path, input_dim, hidden_layers, output_dim, extra_config=None):
    """
    Save DNN model along with its configuration
    """
    ckpt = {
        'model_state_dict': model.state_dict(),
        'config': {
            'input_dim': input_dim,
            'hidden_layers': hidden_layers,
            'output_dim': output_dim,
            **(extra_config or {})
        }
    }
    torch.save(ckpt, path)
    # print(f'DNN model saved to {path}')
    return path
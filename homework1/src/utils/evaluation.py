import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib

def plot_density(df, num_cols=4):
    """
    Plot the density distribution of each numerical feature in the DataFrame.
    """
    num_features = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    num_rows = (len(num_features) + num_cols - 1) // num_cols
    fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(4 * num_cols, 4 * num_rows))
    axes = axes.flatten()

    for i, col in enumerate(num_features):
        # sns.kdeplot(df[col], ax=axes[i], color='red')
        sns.histplot(df[col], ax=axes[i], kde=True, stat='density', bins=30, alpha=0.3)
        axes[i].set_title(f'{col}')

    plt.tight_layout()
    plt.show()

def load_scaler(file_path):
    scaler = joblib.load(file_path)
    return scaler
    
def test_model(model, X, y, is_scaled=False):
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

def plot_loss(train_losses, val_losses, title='', save_path=None, log=False):
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
        plt.title(title + ' (Log Scale)')
    plt.show()

    if save_path:
        plt.savefig(save_path)

def save_results(results, file_path):
    pass
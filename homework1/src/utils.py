import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        print(f"Warning: {e}. Skip for now, trying with 'latin1' encoding.")
        df = pd.read_csv(file_path, encoding='latin1')
        return df

def plot_density(df, num_cols=4):
    """
    Plot the density distribution of each numerical feature in the DataFrame.
    """
    num_features = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    num_rows = (len(num_features) + num_cols - 1) // num_cols
    fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(4 * num_cols, 4 * num_rows))
    axes = axes.flatten()

    for i, col in enumerate(num_features):
        sns.kdeplot(df[col], ax=axes[i])
        axes[i].set_title(f'{col}')

    plt.tight_layout()
    plt.show()

    
def test_model(model, X, y):
    print("\n========= Model Evaluation =========")
    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    
    print(f'Mean Squared Error (MSE): {mse:.4f}')
    print(f'R^2 Score: {r2:.4f}')
    print("="*36)

def plot_loss(train_losses, val_losses, title=''):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('MSE Loss')
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.show()
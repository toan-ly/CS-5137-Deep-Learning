import pandas as pd
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
    
def test_model(model, X, y):
    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    
    print(f'Mean Squared Error (MSE): {mse:.4f}')
    print(f'R^2 Score: {r2:.4f}')

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
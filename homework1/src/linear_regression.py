from sklearn.linear_model import LinearRegression as sklearn_LinearRegression
import numpy as np
import torch
import torch.nn as nn

class LinearRegressionSklearn:
    """
    Linear Regression using sklearn
    """
    def __init__(self):
        self.model = sklearn_LinearRegression()

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)
  

class LinearRegressionNumpy:
    """
    Linear Regression from scratch using NumPy
    """
    def __init__(self, n_features, lr=0.01):
        self.lr = lr
        self.W = np.random.randn(n_features, 1).astype(np.float32) # shape (n_features, 1)
        self.bias = np.zeros(1, dtype=np.float32) # shape (1,)

    def predict(self, X):
        out = X @ self.W + self.bias
        return out

    def loss(self, y_pred, y_true):
        """
        Mean Squared Error Loss (MSE Loss)
        """
        return np.mean((y_pred - y_true) ** 2)

    def gradients(self, X, y, y_pred):
        """
        Compute gradients for weights and bias

        Returns:
            dW: shape (n_features, 1)
            db: shape (1,)
        """
        diff = y_pred - y
        dW = X.T @ diff / len(y)
        db = np.sum(diff) / len(y)
        return dW, db

    def update_weights(self, dW, db):
        self.W -= self.lr * dW
        self.bias -= self.lr * db

    def fit(self, X, y, X_val, y_val, epochs=1000, verbose=True):
        """
        Train model using Gradient Descent

        Args:
            X: shape (n_samples, n_features)
            y: shape (n_samples, 1)
            X_val, y_val: validation set
        """
        train_losses, val_losses = [], []
        for epoch in range(epochs):
            # Forward pass
            y_pred = self.predict(X)
            loss = self.loss(y_pred, y)
            train_losses.append(loss)

            # Backward pass
            dW, db = self.gradients(X, y, y_pred)
            self.update_weights(dW, db)

            # Validation
            y_val_pred = self.predict(X_val)
            val_loss = self.loss(y_val_pred, y_val)
            val_losses.append(val_loss)

            # Logging
            if verbose and (epoch + 1) % int(epochs / 10) == 0:
                print(f'Epoch [{epoch + 1}/{epochs}], Train Loss: {loss:.4f}, Val Loss: {val_loss:.4f}')

        return train_losses, val_losses


class LinearRegressionTorch(nn.Module):
    """
    Linear Regression using PyTorch
    """
    def __init__(self, n_features, lr=0.01):
        super(LinearRegressionTorch, self).__init__()
        self.model = nn.Linear(n_features, 1)
        self.lr = lr
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)

    def fit(
        self, 
        train_loader, 
        val_loader, 
        epochs=1000, 
        verbose=True, 
        early_stopping=True, 
        patience=10, 
        ):
        """
        Train model using Gradient Descent

        Args:
            train_loader: DataLoader for training data
                X: shape (n_samples, n_features)
                y: shape (n_samples, 1)
            val_loader: DataLoader for validation data
                X: shape (n_samples, n_features)
                y: shape (n_samples, 1)
            epochs: number of training epochs
            verbose: whether to print training progress
        """
        train_losses, val_losses = [], []
        best_val_loss = float('inf')
        best_weights = None
        patience_counter = 0
        for epoch in range(epochs):
            # Training loop
            self.model.train()
            batch_train_losses = []
            for X, y in train_loader:
                # Forward pass
                y_pred = self.model(X)

                # Train Loss
                train_loss = self.criterion(y_pred, y)
                batch_train_losses.append(train_loss.item())

                # Backward pass
                self.optimizer.zero_grad()
                train_loss.backward()
                self.optimizer.step()
            train_losses.append(np.mean(batch_train_losses))

            # Validation loss
            self.model.eval()
            batch_val_losses = []
            for X, y in val_loader:
                with torch.no_grad():
                    y_val_pred = self.model(X)
                    val_loss = self.criterion(y_val_pred, y)
                    batch_val_losses.append(val_loss.item())
            val_loss = np.mean(batch_val_losses)
            val_losses.append(val_loss)

            # Logging
            if verbose and (epoch + 1) % int(epochs / 10) == 0:
                print(f'Epoch [{epoch + 1}/{epochs}], Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_loss:.4f}')

            # Early stopping
            if early_stopping:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_weights = self.model.state_dict()
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f'Early stopping at epoch {epoch + 1}')
                        if best_weights is not None:
                            self.model.load_state_dict(best_weights) # Restore best weights
                        break

        return train_losses, val_losses

    def predict(self, X):   
        self.model.eval()
        with torch.no_grad():
            y_pred = self.model(X)
        return y_pred.cpu().numpy() 
 
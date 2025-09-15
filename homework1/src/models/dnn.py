import torch
import torch.nn as nn
import numpy as np


class DNN(nn.Module):
    """
    Deep Neural Network (DNN).
    """
    def __init__(
        self, 
        input_dim: int, 
        hidden_layers: list, 
        output_dim: int = 1, 
        dropout: float = 0.0,
        batch_norm: bool = False,
        lr: float = 0.01,
        l2_lambda: float = 0.0,
        activation='sigmoid',
    ):
        """
        Args:
            input_dim: number of input features
            hidden_layers: list of hidden layer sizes, e.g. [16, 32]
            output_dim: number of output features
            lr: learning rate
        """
        super(DNN, self).__init__()

        self.activation = activation
        if activation == 'relu':
            self.act = nn.ReLU()
        elif activation == 'tanh':
            self.act = nn.Tanh()
        elif activation == 'sigmoid':
            self.act = nn.Sigmoid()
        elif activation == 'leaky_relu':
            self.act = nn.LeakyReLU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        layers = []
        layers.append(nn.Linear(input_dim, hidden_layers[0]))
        layers.append(self.act)
        for i in range(len(hidden_layers) - 1):
            layers.append(nn.Linear(hidden_layers[i], hidden_layers[i + 1]))

            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_layers[i + 1]))

            layers.append(self.act)

            if dropout > 0:
                layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(hidden_layers[-1], output_dim))
        self.model = nn.Sequential(*layers)
        self.init_weights()

        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.parameters(), lr=lr, weight_decay=1e-4)
        self.lr = lr
        self.l2_lambda = l2_lambda

    def init_weights(self):
        for layer in self.model:
            if isinstance(layer, nn.Linear):
                if self.activation == 'relu':
                    nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
                elif self.activation == 'sigmoid':
                    nn.init.xavier_normal_(layer.weight)
                else:
                    nn.init.normal_(layer.weight, mean=0.0, std=0.01)
                nn.init.zeros_(layer.bias)
    
    def forward(self, x):
        return self.model(x)

    def fit(
        self, 
        train_loader, 
        val_loader,
        epochs=1000, 
        verbose=True, 
        early_stopping=False, 
        patience=10, 
        grad_clip=False,
        ):
        """
        Train model using SGD optimizer

        Args:
            train_loader: DataLoader for training data
                X: shape (n_samples, n_features)
                y: shape (n_samples, 1)
            val_loader: DataLoader for validation data
                X: shape (n_samples, n_features)
                y: shape (n_samples, 1)
            epochs: number of training epochs
            verbose: whether to print training progress
            early_stopping: whether to use early stopping
            patience: number of epochs to wait for improvement before stopping
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
                self.optimizer.zero_grad()

                y_pred = self.forward(X)    
                train_loss = self.criterion(y_pred, y)

                # L2 regularization
                if self.l2_lambda > 0:
                    l2_loss = 0
                    for param in self.parameters():
                        l2_loss += torch.sum(param ** 2)
                    train_loss += self.l2_lambda * l2_loss

                batch_train_losses.append(train_loss.item())

                # Backward pass
                train_loss.backward()

                # Gradient clipping to prevent exploding gradients
                if grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=10.0)
                self.optimizer.step()

            train_loss = np.mean(batch_train_losses)
            train_losses.append(train_loss)

            # Validation loop
            self.model.eval()
            batch_val_losses = []
            with torch.no_grad():
                for X_val, y_val in val_loader:
                    y_val_pred = self.forward(X_val)
                    val_loss = self.criterion(y_val_pred, y_val)
                    batch_val_losses.append(val_loss.item())
            val_loss = np.mean(batch_val_losses)
            val_losses.append(val_loss)

            # Logging
            if verbose and (epoch + 1) % (epochs // 10) == 0:
                print(f'Epoch [{epoch + 1}/{epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

            # Early stopping
            if early_stopping:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_weights = self.state_dict()
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f'Early stopping at epoch {epoch + 1}')
                        if best_weights is not None:
                            self.load_state_dict(best_weights) # Restore best weights
                        break

        return train_losses, val_losses

    def predict(self, X):
        self.eval()
        with torch.no_grad():
            y_pred = self.forward(X)
        return y_pred.cpu().numpy()
import torch
import torch.nn as nn

class DNN(nn.Module):
    def __init__(self, input_dim: int, hidden_layers: list, output_dim: int = 1, lr: float = 0.01):
        super(DNN, self).__init__()
        layers = []

        layers.append(nn.Linear(input_dim, hidden_layers[0]))
        layers.append(nn.ReLU())
        for i in range(len(hidden_layers) - 1):
            layers.append(nn.Linear(hidden_layers[i], hidden_layers[i + 1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_layers[-1], output_dim))
        self.model = nn.Sequential(*layers)
        # self.init_weights()

        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.parameters(), lr=lr)

    def init_weights(self):
        for layer in self.model:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
    
    def forward(self, x):
        return self.model(x)
    
    def fit(self, X, y, X_val, y_val, epochs=1000, verbose=True):
        """
        Train model using SGD optimizer

        Args:
            X: shape (n_samples, n_features)
            y: shape (n_samples, 1)
            X_val, y_val: validation set
        """
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
        X_val = torch.tensor(X_val, dtype=torch.float32)
        y_val = torch.tensor(y_val, dtype=torch.float32)

        train_losses, val_losses = [], []
        for epoch in range(epochs):
            # Forward pass
            self.optimizer.zero_grad()

            y_pred = self.forward(X)    
            train_loss = self.criterion(y_pred, y)
            train_losses.append(train_loss.item())

            # Validation loss
            with torch.no_grad():
                y_val_pred = self.forward(X_val)
                val_loss = self.criterion(y_val_pred, y_val)
                val_losses.append(val_loss.item())

            # Backward pass
            train_loss.backward()
            self.optimizer.step()

            if verbose and epoch % 100 == 0:
                print(f"Epoch {epoch}: Train loss: {train_loss.item()}, Val loss: {val_loss.item()}")

        return train_losses, val_losses
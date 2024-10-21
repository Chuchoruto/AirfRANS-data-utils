import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

class AirFNN(nn.Module):
    """
    Neural Network for predicting v_x, v_y, and sdf given x and y positions.
    """
    def __init__(self):
        super(AirFNN, self).__init__()
        # Define the layers of the network
        self.model = nn.Sequential(
            nn.Linear(2, 16),  # Input: (x, y), Output: 64 hidden units
            nn.ReLU(),          # Activation
            nn.Linear(16, 16), # Hidden layer
            nn.ReLU(),          # Activation
            nn.Linear(16, 16), # Hidden layer
            nn.ReLU(),          # Activation
            nn.Linear(16, 3)    # Output: (v_x, v_y, sdf)
        )
    def forward(self, x):
        return self.model(x)
def train_model(X_tensor, Y_tensor, model, epochs=20, batch_size=64, learning_rate=0.001):
    """
    Trains the given model on the input and target tensors.
    
    Parameters:
        X_tensor (torch.Tensor): Input tensor (x, y positions).
        Y_tensor (torch.Tensor): Target tensor (v_x, v_y, sdf).
        model (nn.Module): PyTorch model to train.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
        learning_rate (float): Learning rate for the optimizer.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Using device: {device}")

    # Move tensors to GPU
    X_tensor, Y_tensor = X_tensor.to(device), Y_tensor.to(device)

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(epochs):
        model.train()  # Set the model to training mode
        total_loss = 0.0

        # Shuffle data at the start of each epoch
        indices = torch.randperm(X_tensor.size(0))
        X_tensor = X_tensor[indices]
        Y_tensor = Y_tensor[indices]

        # Iterate over batches
        for i in range(0, X_tensor.size(0), batch_size):
            X_batch = X_tensor[i:i + batch_size]
            Y_batch = Y_tensor[i:i + batch_size]

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(X_batch)
            loss = criterion(outputs, Y_batch)

            # Backward pass and optimization step
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            print("Batch complete")

        # Print the average loss for the epoch
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / (X_tensor.size(0) / batch_size):.4f}")

    print("Training complete!")
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
def train_model(dataset, model, epochs=20, batch_size=512, learning_rate=0.001):
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

    # Create DataLoader with optimizations
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, 
        num_workers=4, pin_memory=True
    )

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Initialize mixed precision scaler
    scaler = torch.cuda.amp.GradScaler()

    # Training loop
    for epoch in range(epochs):
        model.train()  # Set the model to training mode
        total_loss = 0.0

        for X_batch, Y_batch in dataloader:
            # Move data to GPU
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)

            # Zero the gradients from the previous iteration
            optimizer.zero_grad()

            # Forward pass with mixed precision
            with torch.cuda.amp.autocast():
                outputs = model(X_batch)
                loss = criterion(outputs, Y_batch)

            # Backward pass with scaling
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

        # Print the average loss for the epoch
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(dataloader):.4f}")

    print("Training complete!")

    
    
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from data_manip.manip_utils import ChunkedDataset

class AirFNN(nn.Module):
    """
    Neural Network for predicting v_x, v_y, and sdf given x and y positions.
    """

    def __init__(self):
        super(AirFNN, self).__init__()
        # Define the layers of the network
        self.model = nn.Sequential(
            nn.Linear(2, 64),  # Input: (x, y), Output: 64 hidden units
            nn.ReLU(),          # Activation
            nn.Linear(64, 128), # Hidden layer
            nn.ReLU(),          # Activation
            nn.Linear(128, 64), # Hidden layer
            nn.ReLU(),          # Activation
            nn.Linear(64, 3)    # Output: (v_x, v_y, sdf)
        )

    def forward(self, x):
        return self.model(x)

def train_model(model, dataset_dir, epochs=20, batch_size=64, learning_rate=0.001):
    """
    Trains the given model on the chunked dataset.

    Parameters:
        model (nn.Module): PyTorch model to train.
        dataset_dir (str): Directory containing the chunked dataset.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
        learning_rate (float): Learning rate for the optimizer.
    """
    # Create a ChunkedDataset
    dataset = ChunkedDataset(dataset_dir, chunk_prefix='chunk_')

    # Create a DataLoader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Using device: {device}")

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(epochs):
        model.train()  # Set the model to training mode
        total_loss = 0

        for batch in dataloader:
            # Separate features and targets
            X = batch[:, :2].to(device)  # First two columns are x, y
            y = batch[:, 2:].to(device)  # Last three columns are v_x, v_y, sdf

            # Zero the gradients from the previous iteration
            optimizer.zero_grad()

            # Forward pass
            outputs = model(X)

            # Compute the loss
            loss = criterion(outputs, y)

            # Backward pass and optimization step
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # Print the loss for every epoch
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

    print("Training complete!")



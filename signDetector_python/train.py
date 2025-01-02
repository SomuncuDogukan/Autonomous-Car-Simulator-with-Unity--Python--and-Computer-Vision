import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

# Dataset loading and preprocessing
data_dir = "dataset"  # Path to the dataset

transform = transforms.Compose([
    transforms.Resize((64, 64)),  # Resize images to 64x64
    transforms.ToTensor(),  # Convert images to tensors
    transforms.Normalize((0.5,), (0.5,))  # Normalize the images
])

# Load the dataset from the directory
dataset = datasets.ImageFolder(data_dir, transform=transform)

# Split the dataset into training and testing sets (80% train, 20% test)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# DataLoader to handle batching and shuffling
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Model definition
class TrafficSignNet(nn.Module):
    def __init__(self):
        super(TrafficSignNet, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3),  # Convolutional layer 1
            nn.ReLU(),  # ReLU activation function
            nn.MaxPool2d(2, 2),  # Max pooling layer 1
            nn.Conv2d(32, 64, kernel_size=3),  # Convolutional layer 2
            nn.ReLU(),  # ReLU activation function
            nn.MaxPool2d(2, 2)  # Max pooling layer 2
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),  # Flatten the image for fully connected layers
            nn.Linear(64 * 14 * 14, 128),  # Fully connected layer 1
            nn.ReLU(),  # ReLU activation function
            nn.Linear(128, 2)  # Output layer with 2 classes (e.g., speed limit signs)
        )

    def forward(self, x):
        x = self.conv_layers(x)  # Apply convolutional layers
        x = self.fc_layers(x)  # Apply fully connected layers
        return x

# Initialize the model, loss function, and optimizer
model = TrafficSignNet()
criterion = nn.CrossEntropyLoss()  # Loss function
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Optimizer

# Model training function
def train_model(model, loader, criterion, optimizer, epochs=10):
    model.train()  # Set the model to training mode
    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in loader:
            optimizer.zero_grad()  # Zero the gradients
            outputs = model(images)  # Forward pass
            loss = criterion(outputs, labels)  # Compute the loss
            loss.backward()  # Backward pass
            optimizer.step()  # Update the model parameters
            running_loss += loss.item()  # Track the loss
        print(f"Epoch {epoch+1}, Loss: {running_loss/len(loader)}")  # Log the loss for each epoch

# Train the model
train_model(model, train_loader, criterion, optimizer)

# Save the trained model
torch.save(model.state_dict(), "traffic_sign_model.pth")
print("Model saved to traffic_sign_model.pth")  # Confirmation message when model is saved

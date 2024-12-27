import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Hyperparameters
# Batch training because computing gradients over the entire dataset is computationally expensive
# as we compute average gradient over all samples
batch_size = 64
learning_rate = 0.001
epochs = 10

if torch.backends.mps.is_available():
    device = torch.device('mps')  # Use MPS for Apple Silicon
elif torch.cuda.is_available():
    device = torch.device('cuda')  # Use CUDA for NVIDIA GPUs
else:
    device = torch.device('cpu')  # Fallback to CPU

print(f"Using device: {device}")

# Data transformations
transform = transforms.Compose([
    # converts images to PyTorch tensors and scales pixel values to [0, 1].
    transforms.ToTensor(),
    # normalise using mean = 0.5 and st_dev = 0.5 with z-score formula (x - mean) / st_dev
    # basically, x_normalised = 2x - 1 and remember all x's from ToTensor() are [0, 1]
    # so we normalise all pixel values from [-1, 1].

    # We do this because many activation functions (like tanh or ReLU) and optimization 
    # algorithms perform better when the input data is normalized and has a mean close to 0.
    transforms.Normalize((0.5,), (0.5,))
])

# Load Fashion MNIST dataset
train_dataset = datasets.FashionMNIST(
    root='./data', train=True, transform=transform, download=True
)
test_dataset = datasets.FashionMNIST(
    root='./data', train=False, transform=transform, download=True
)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Make the Neural Network
class FashionMNISTClassifier(nn.Module):
    def __init__(self):
        super(FashionMNISTClassifier, self).__init__()
        # Define layers explicitly
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, 256)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(256, 128)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(128, 10)
    
    def forward(self, x):
        # Define the forward pass
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x

# Instantiate the model, loss function, and optimizer - what actually updates weights
model = FashionMNISTClassifier().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
def train(model, loader, criterion, optimizer):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for imgs, targets in loader:
            imgs, targets = imgs.to(device), targets.to(device)

            # Forward pass
            outputs = model(imgs)
            loss = criterion(outputs, targets)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(loader):.4f}")

# Evaluation function
def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Accuracy: {accuracy:.2f}%")

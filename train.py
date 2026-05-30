import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os

# ─────────────────────────────────────────
# 1. LOAD THE MNIST DATASET
# ─────────────────────────────────────────
transform = transforms.ToTensor()

train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_data  = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_data,  batch_size=64, shuffle=False)

# ─────────────────────────────────────────
# 2. DEFINE THE NEURAL NETWORK
# ─────────────────────────────────────────
class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)   # Input layer  → Hidden layer 1
        self.fc2 = nn.Linear(128, 64)       # Hidden layer 1 → Hidden layer 2
        self.fc3 = nn.Linear(64, 10)        # Hidden layer 2 → Output (10 digits)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(-1, 28*28)              # Flatten image to 1D
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = NeuralNet()

# ─────────────────────────────────────────
# 3. LOSS FUNCTION & OPTIMIZER
# ──────────────────────────
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ─────────────────────────────────────────
# 4. TRAIN THE MODEL
# ─────────────────────────────────────────
train_losses = []
train_accuracies = []
EPOCHS = 10

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / len(train_loader)
    accuracy = 100 * correct / total
    train_losses.append(avg_loss)
    train_accuracies.append(accuracy)

    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f} | Accuracy: {accuracy:.2f}%")

# ─────────────────────────────────────────
# 5. SAVE TRAINED MODEL
# ─────────────────────────────────────────
torch.save(model.state_dict(), 'model.pth')
print("\n✅ Model saved as model.pth")

# ─────────────────────────────────────────
# 6. SAVE LOSS & ACCURACY FOR VISUALIZATION
# ─────────────────────────────────────────
import json
os.makedirs('plots', exist_ok=True)
with open('plots/metrics.json', 'w') as f:
    json.dump({'losses': train_losses, 'accuracies': train_accuracies}, f)

print("✅ Metrics saved to plots/metrics.json")
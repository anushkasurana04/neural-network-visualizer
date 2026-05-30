import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import json
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ─────────────────────────────────────────
# LOAD SAVED METRICS
# ─────────────────────────────────────────
with open('plots/metrics.json', 'r') as f:
    metrics = json.load(f)

train_losses     = metrics['losses']
train_accuracies = metrics['accuracies']
epochs           = list(range(1, len(train_losses) + 1))

# ─────────────────────────────────────────
# REBUILD MODEL & LOAD WEIGHTS
# ─────────────────────────────────────────
class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = NeuralNet()
model.load_state_dict(torch.load('model.pth'))
model.eval()

# ─────────────────────────────────────────
# PLOT 1 — LOSS CURVE
# ─────────────────────────────────────────
plt.figure(figsize=(8, 5))
plt.plot(epochs, train_losses, marker='o', color='tomato', linewidth=2)
plt.title('Training Loss per Epoch', fontsize=14, fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.xticks(epochs)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('plots/loss_curve.png')
plt.show()
print("✅ Loss curve saved!")

# ─────────────────────────────────────────
# PLOT 2 — ACCURACY CURVE
# ─────────────────────────────────────────
plt.figure(figsize=(8, 5))
plt.plot(epochs, train_accuracies, marker='s', color='steelblue', linewidth=2)
plt.title('Training Accuracy per Epoch', fontsize=14, fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.xticks(epochs)
plt.ylim([80, 101])
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('plots/accuracy_curve.png')
plt.show()
print("✅ Accuracy curve saved!")

# ─────────────────────────────────────────
# PLOT 3 — SAMPLE PREDICTIONS
# ─────────────────────────────────────────
transform   = transforms.ToTensor()
test_data   = datasets.MNIST(root='./data', train=False, download=False, transform=transform)
test_loader = DataLoader(test_data, batch_size=16, shuffle=True)

images, labels = next(iter(test_loader))

with torch.no_grad():
    outputs   = model(images)
    _, predicted = torch.max(outputs, 1)

fig, axes = plt.subplots(2, 8, figsize=(16, 5))
fig.suptitle('Sample Predictions (Green = Correct, Red = Wrong)', fontsize=13, fontweight='bold')

for i, ax in enumerate(axes.flat):
    img   = images[i].squeeze().numpy()
    label = labels[i].item()
    pred  = predicted[i].item()
    color = 'green' if pred == label else 'red'

    ax.imshow(img, cmap='gray')
    ax.set_title(f'P:{pred} A:{label}', color=color, fontsize=9)
    ax.axis('off')

plt.tight_layout()
plt.savefig('plots/sample_predictions.png')
plt.show()
print("✅ Sample predictions saved!")

# ─────────────────────────────────────────
# PLOT 4 — CONFUSION MATRIX
# ─────────────────────────────────────────
from sklearn.metrics import confusion_matrix
import seaborn as sns

all_preds  = []
all_labels = []

full_test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

with torch.no_grad():
    for images, labels in full_test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.numpy())
        all_labels.extend(labels.numpy())

cm = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(9, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=range(10), yticklabels=range(10))
plt.title('Confusion Matrix — Which digits get confused?', fontsize=13, fontweight='bold')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.savefig('plots/confusion_matrix.png')
plt.show()
print("✅ Confusion matrix saved!")

print("\n🎉 All 4 visualizations saved in the plots/ folder!")
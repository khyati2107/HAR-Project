import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

os.makedirs('outputs', exist_ok=True)


class HAR_Dataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class HAR_MLP(nn.Module):
    def __init__(self, input_dim, hidden1=128, hidden2=64, num_classes=6):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, num_classes)
        )

    def forward(self, x):
        return self.model(x)


def run_neural_network(processed_data, epochs=30, batch_size=32):
    X_train = torch.tensor(processed_data['X_train'], dtype=torch.float32)
    X_test  = torch.tensor(processed_data['X_test'],  dtype=torch.float32)
    y_train = torch.tensor(processed_data['y_train'].values - 1, dtype=torch.long)
    y_test  = torch.tensor(processed_data['y_test'].values  - 1, dtype=torch.long)

    num_features = X_train.shape[1]
    num_classes  = len(torch.unique(y_train))

    train_loader = DataLoader(
        HAR_Dataset(X_train, y_train),
        batch_size=batch_size,
        shuffle=True
    )

    model     = HAR_MLP(num_features, num_classes=num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train_losses = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")

    model.eval()
    with torch.no_grad():
        y_pred = torch.argmax(model(X_test), dim=1)

    acc = accuracy_score(y_test.numpy(), y_pred.numpy())
    print(f"\nTest Accuracy: {acc*100:.2f}%")
    print("\nClassification Report:\n",
          classification_report(y_test.numpy(), y_pred.numpy(), digits=3))

    # Loss plot
    plt.figure(figsize=(8, 4))
    plt.plot(train_losses, marker='o', markersize=3)
    plt.title('Neural Network Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('outputs/neural_network_training_loss.png', dpi=150, bbox_inches='tight')
    plt.show()

    # Confusion matrix
    cm = confusion_matrix(y_test.numpy(), y_pred.numpy())
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix - MLP HAR")
    plt.tight_layout()
    plt.savefig('outputs/neural_network_confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.show()
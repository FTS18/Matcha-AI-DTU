import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

def train_action_classifier(data_dir, epochs=10, batch_size=32):
 """
 Train a simple image classifier on top of ResNet18.
 Data directory should contain subfolders as classes (e.g., 'goal', 'foul', 'foul_risk').
 """
 print(f" Loading data from {data_dir}...")
 
 # Image transformations
 transform = transforms.Compose([
 transforms.Resize((224, 224)),
 transforms.ToTensor(),
 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
 ])
 
 dataset = datasets.ImageFolder(data_dir, transform=transform)
 loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
 class_names = dataset.classes
 num_classes = len(class_names)
 
 print(f" Detected {num_classes} classes: {class_names}")
 
 # Load Pretrained ResNet18
 model = models.resnet18(pretrained=True)
 # Freeze layers
 for param in model.parameters():
 param.requires_grad = False
 
 # Replace final layer
 model.fc = nn.Linear(model.fc.in_features, num_classes)
 
 device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 model = model.to(device)
 
 criterion = nn.CrossEntropyLoss()
 optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
 
 print(" Starting training...")
 for epoch in range(epochs):
 model.train()
 running_loss = 0.0
 for inputs, labels in loader:
 inputs, labels = inputs.to(device), labels.to(device)
 
 optimizer.zero_grad()
 outputs = model(inputs)
 loss = criterion(outputs, labels)
 loss.backward()
 optimizer.step()
 
 running_loss += loss.item() * inputs.size(0)
 
 epoch_loss = running_loss / len(dataset)
 print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4)f}")
 
 # Save the model
 os.makedirs("models", exist_ok=True)
 torch.save(model.state_dict(), "models/action_classifier.pth")
 
 # Save metadata
 with open("models/metadata.json", "w") as f:
 import json
 json.dump({"classes": class_names}, f)
 
 print(f" Model saved to models/action_classifier.pth")

if __name__ == "__main__":
 import argparse
 parser = argparse.ArgumentParser(description="Train custom action spotter")
 parser.add_argument("--data", type=str, required=True, help="Path to organized image folders")
 parser.add_argument("--epochs", type=int, default=10)
 
 args = parser.parse_args()
 train_action_classifier(args.data, args.epochs)

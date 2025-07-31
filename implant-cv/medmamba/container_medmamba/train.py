import argparse
import os
import torch
from MedMamba import MedMamba  # make sure this file is in the same directory
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
import os

def get_num_classes(data_dir):
    classes = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    return len(classes)


def train(args):
    print(f"🧠 Starting MedMamba training for {args.epochs} epochs...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MedMamba(num_classes=2).to(device)

    dataset = DummyDataset(args.data_dir)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}/{args.epochs}, Loss: {running_loss:.4f}")

    # Save model (SageMaker will copy from this directory to S3)
    model_path = os.path.join(args.model_dir, "model.pth")
    torch.save(model.state_dict(), model_path)
    print(f"✅ Saved trained model to {model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--model_dir", type=str, default="/opt/ml/model")
    parser.add_argument("--data_dir", type=str, default="/opt/ml/input/data/train")
    args = parser.parse_args()

    train(args)

import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, transforms
from sklearn.preprocessing import MultiLabelBinarizer
import joblib

from dataset import ImplantDataset

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_path', type=str, required=True)
    parser.add_argument('--image_root', type=str, required=True)
    parser.add_argument('--model-dir', type=str, default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-3)
    return parser.parse_args()

def main():
    args = parse_args()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    dataset = ImplantDataset(args.json_path, args.image_root, transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Label binarization
    all_labels = [lbl for _, labels in dataset.samples for lbl in labels]
    classes = sorted(set(all_labels))
    mlb = MultiLabelBinarizer(classes=classes)
    Y = mlb.fit_transform([labels for _, labels in dataset.samples])

    model = models.resnet18(pretrained=True)
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, len(classes)),
        nn.Sigmoid()
    )

    model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.BCELoss()

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0

        for i, (images, labels) in enumerate(loader):
            labels = mlb.transform(labels)
            labels = torch.tensor(labels, dtype=torch.float32)

            images, labels = images.to(model.device), labels.to(model.device)
            preds = model(images)
            loss = loss_fn(preds, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{args.epochs} - Loss: {total_loss/len(loader):.4f}")

    # Save model and label binarizer
    os.makedirs(args.model_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(args.model_dir, "model.pth"))
    joblib.dump(mlb, os.path.join(args.model_dir, "label_binarizer.joblib"))

if __name__ == "__main__":
    main()


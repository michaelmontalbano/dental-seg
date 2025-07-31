import torch
import torch.nn as nn
import torchvision.models as models
from transformers import ViTModel

class CNNTransformerHybrid(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.cnn = models.resnet50(pretrained=True)
        self.cnn.fc = nn.Identity()
        self.transformer = ViTModel.from_pretrained('google/vit-base-patch16-224')
        self.classifier = nn.Linear(2048 + 768, num_classes)

    def forward(self, x):
        cnn_features = self.cnn(x)
        vit_features = self.transformer(pixel_values=x).last_hidden_state.mean(1)
        combined = torch.cat([cnn_features, vit_features], dim=1)
        return self.classifier(combined)

import torch
from torch import nn
from torchvision.models import ResNet50_Weights, resnet50

LABELS = ["authentic", "tampered"]


def build_model(pretrained: bool = True) -> nn.Module:
    weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    model = resnet50(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, len(LABELS))
    return model


def load_model(checkpoint_path, device: str = "cpu") -> nn.Module:
    model = build_model(pretrained=False)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model

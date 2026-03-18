from __future__ import annotations

import argparse
from typing import Tuple, Dict, Any

import torch
from PIL import Image
from torchvision import models, transforms


def load_model(checkpoint_path: str) -> Tuple[torch.nn.Module, Dict[str, int]]:
    """
    Load a trained ResNet18 classifier and class_to_idx mapping.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint: Dict[str, Any] = torch.load(checkpoint_path, map_location=device)

    class_to_idx = checkpoint["class_to_idx"]
    model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = torch.nn.Linear(in_features, len(class_to_idx))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, class_to_idx


def preprocess_image(image_path: str) -> torch.Tensor:
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)


def predict_image_class(checkpoint_path: str, image_path: str) -> str:
    model, class_to_idx = load_model(checkpoint_path)
    device = next(model.parameters()).device

    tensor = preprocess_image(image_path).to(device)
    with torch.no_grad():
        outputs = model(tensor)
        _, pred = torch.max(outputs, 1)

    idx_to_class = {idx: cls_name for cls_name, idx in class_to_idx.items()}
    return idx_to_class[int(pred.item())]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run image classification inference.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained model checkpoint (.pt).")
    parser.add_argument("--image", type=str, required=True, help="Path to input image.")
    args = parser.parse_args()

    label = predict_image_class(args.checkpoint, args.image)
    print(label)


if __name__ == "__main__":
    main()


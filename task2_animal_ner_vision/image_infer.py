from __future__ import annotations

import argparse
from typing import Tuple, Dict, Any
import os

import torch
from PIL import Image
from torchvision import models, transforms
from huggingface_hub import hf_hub_download


DEFAULT_REPO = "hesoyam3333/test_task_winstars"
DEFAULT_MODEL_PATH_IN_REPO = "image_model/model.pt"


def resolve_checkpoint_path(checkpoint: str | None) -> str:
    """
    Resolve checkpoint path:
    - if local path exists -> use it
    - otherwise -> download from Hugging Face
    """
    if checkpoint and os.path.exists(checkpoint):
        return checkpoint

    print("⬇️ Downloading model from Hugging Face...")
    path = hf_hub_download(
        repo_id=DEFAULT_REPO,
        filename=DEFAULT_MODEL_PATH_IN_REPO,
    )
    return path


def load_model(checkpoint_path: str) -> Tuple[torch.nn.Module, Dict[str, int]]:
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


def predict_image_class(checkpoint: str | None, image_path: str) -> str:
    checkpoint_path = resolve_checkpoint_path(checkpoint)

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
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Local path to model (.pt). If not provided, downloads from Hugging Face.",
    )
    parser.add_argument("--image", type=str, required=True)
    args = parser.parse_args()

    label = predict_image_class(args.checkpoint, args.image)
    print(label)


if __name__ == "__main__":
    main()
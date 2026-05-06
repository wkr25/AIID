from __future__ import annotations

import sys
from pathlib import Path

import torch
from PIL import Image

from dataset import Transform
from model import Detector


def load_model(model_path: str | Path = "detector.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Detector(num_class=2)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model, device


@torch.no_grad()
def predict_image(image_path: str | Path, model_path: str | Path = "detector.pth") -> dict[str, float | str]:
    model, device = load_model(model_path)
    transform = Transform(image_size=64)
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    logit, _ = model(image)
    probs = torch.softmax(logit, dim=1)[0]

    fake_prob = float(probs[0].item())
    real_prob = float(probs[1].item())
    prediction = "fake" if fake_prob >= real_prob else "real"
    ai_degree = fake_prob

    return {
        "image_path": str(image_path),
        "prediction": prediction,
        "ai_degree": ai_degree,
        "real_degree": real_prob,
        "fake_prob": fake_prob,
        "real_prob": real_prob,
    }


def print_prediction(result: dict[str, float | str]) -> None:
    print(f"image: {result['image_path']}")
    print(f"AI degree: {result['ai_degree'] * 100:.2f}%")
    print(f"real degree: {result['real_degree'] * 100:.2f}%")
    print(f"prediction: {result['prediction']}")


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python predict.py path/to/image.jpg")
        return
    result = predict_image(sys.argv[1])
    print_prediction(result)


if __name__ == "__main__":
    main()

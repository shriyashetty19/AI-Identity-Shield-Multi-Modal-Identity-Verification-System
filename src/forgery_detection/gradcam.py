"""Grad-CAM visualization for the forgery detector: highlights which region
of a document the model used to call it tampered vs. authentic."""
import argparse
import csv

import numpy as np
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from src.common.config import MIDV2020_TAMPER_DIR, MODELS_DIR, OUTPUTS_DIR, ROOT
from src.forgery_detection.dataset import IMAGE_SIZE, build_transforms
from src.forgery_detection.forensics import compute_ela
from src.forgery_detection.model import load_model


def run_gradcam(checkpoint_path, split: str, n_samples: int, out_dir, preprocessing: str = "ela") -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(checkpoint_path, device)
    target_layer = model.layer4[-1]
    cam = GradCAM(model=model, target_layers=[target_layer])

    manifest = MIDV2020_TAMPER_DIR / f"{split}_manifest.csv"
    with open(manifest, newline="") as f:
        rows = list(csv.DictReader(f))

    rng = np.random.default_rng(0)
    sample_idx = rng.choice(len(rows), size=min(n_samples, len(rows)), replace=False)

    transform = build_transforms(train=False, preprocessing=preprocessing)
    out_dir.mkdir(parents=True, exist_ok=True)

    for i in sample_idx:
        row = rows[i]
        image = Image.open(ROOT / row["filepath"]).convert("RGB")
        input_tensor = transform(image).unsqueeze(0).to(device)

        label = int(row["label"])
        grayscale_cam = cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(label)])[0]

        # Overlay on whatever the model actually sees (ELA map or raw photo),
        # not always the raw photo, so the heatmap can be checked against the
        # visible artifact.
        if preprocessing == "ela":
            display_img = compute_ela(image).resize((IMAGE_SIZE, IMAGE_SIZE))
        else:
            display_img = image.resize((IMAGE_SIZE, IMAGE_SIZE))
        rgb_img = np.array(display_img).astype(np.float32) / 255.0
        overlay = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

        out_path = out_dir / f"gradcam_{i}_{row['technique'].replace(':', '-')}.png"
        Image.fromarray(overlay).save(out_path)

    print(f"saved {len(sample_idx)} Grad-CAM overlays to {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=str(MODELS_DIR / "forgery_resnet50.pt"))
    parser.add_argument("--split", default="valid")
    parser.add_argument("--n-samples", type=int, default=8)
    parser.add_argument("--preprocessing", choices=["ela", "raw"], default="ela")
    args = parser.parse_args()
    run_gradcam(args.checkpoint, args.split, args.n_samples, OUTPUTS_DIR / "gradcam", args.preprocessing)


if __name__ == "__main__":
    main()

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from torch.utils.data import DataLoader

from src.common.config import MIDV2020_TAMPER_DIR, MODELS_DIR, OUTPUTS_DIR
from src.forgery_detection.dataset import ForgeryDataset
from src.forgery_detection.model import LABELS, load_model


@torch.no_grad()
def evaluate(model, loader: DataLoader, device: str) -> dict:
    all_preds, all_labels = [], []
    for images, labels in loader:
        images = images.to(device)
        logits = model(images)
        preds = logits.argmax(dim=1).cpu()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.tolist())

    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="binary", zero_division=0
    )
    accuracy = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    cm = confusion_matrix(all_labels, all_preds)
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm,
    }


def save_confusion_matrix(cm, out_path) -> None:
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(LABELS)))
    ax.set_yticks(range(len(LABELS)))
    ax.set_xticklabels(LABELS)
    ax.set_yticklabels(LABELS)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=str(MODELS_DIR / "forgery_resnet50.pt"))
    parser.add_argument("--split", default="valid")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--preprocessing", choices=["ela", "raw"], default="ela")
    parser.add_argument("--data-dir", default=None, help="defaults to data/midv2020_tamper")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args.checkpoint, device)

    data_dir = Path(args.data_dir) if args.data_dir else MIDV2020_TAMPER_DIR
    manifest = data_dir / f"{args.split}_manifest.csv"
    dataset = ForgeryDataset(manifest, train=False, preprocessing=args.preprocessing)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    metrics = evaluate(model, loader, device)
    print(f"accuracy:  {metrics['accuracy']:.4f}")
    print(f"precision: {metrics['precision']:.4f}")
    print(f"recall:    {metrics['recall']:.4f}")
    print(f"f1:        {metrics['f1']:.4f}")
    print("confusion matrix:")
    print(metrics["confusion_matrix"])

    out_path = OUTPUTS_DIR / "confusion_matrix.png"
    save_confusion_matrix(metrics["confusion_matrix"], out_path)
    print(f"saved confusion matrix plot to {out_path}")


if __name__ == "__main__":
    main()

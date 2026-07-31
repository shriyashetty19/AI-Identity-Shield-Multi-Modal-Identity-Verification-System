import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.common.config import MIDV2020_TAMPER_DIR, MODELS_DIR
from src.forgery_detection.dataset import ForgeryDataset
from src.forgery_detection.evaluate import evaluate
from src.forgery_detection.model import build_model


def train(
    epochs: int = 8,
    batch_size: int = 16,
    lr: float = 1e-4,
    preprocessing: str = "ela",
    checkpoint_path=None,
    data_dir=None,
) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}  preprocessing: {preprocessing}", flush=True)

    data_dir = data_dir or MIDV2020_TAMPER_DIR
    train_manifest = data_dir / "train_manifest.csv"
    valid_manifest = data_dir / "valid_manifest.csv"
    if not train_manifest.exists():
        raise SystemExit(
            "train_manifest.csv not found - run scripts/generate_synthetic_tampering.py first."
        )

    train_loader = DataLoader(
        ForgeryDataset(train_manifest, train=True, preprocessing=preprocessing),
        batch_size=batch_size,
        shuffle=True,
    )
    valid_loader = DataLoader(
        ForgeryDataset(valid_manifest, train=False, preprocessing=preprocessing),
        batch_size=batch_size,
        shuffle=False,
    )
    print(f"{len(train_loader)} batches/epoch", flush=True)

    model = build_model(pretrained=True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    checkpoint_path = checkpoint_path or MODELS_DIR / "forgery_resnet50.pt"
    best_f1 = -1.0

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for step, (images, labels) in enumerate(train_loader, start=1):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            if step % 100 == 0:
                print(f"  epoch {epoch}/{epochs} step {step}/{len(train_loader)}", flush=True)

        train_loss = running_loss / len(train_loader.dataset)
        scheduler.step()

        model.eval()
        metrics = evaluate(model, valid_loader, device)
        print(
            f"epoch {epoch}/{epochs}  train_loss={train_loss:.4f}  "
            f"val_acc={metrics['accuracy']:.4f}  val_f1={metrics['f1']:.4f}",
            flush=True,
        )

        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  -> saved new best checkpoint (f1={best_f1:.4f}) to {checkpoint_path}", flush=True)

    print(f"training complete. best val f1 = {best_f1:.4f}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--preprocessing", choices=["ela", "raw"], default="ela")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--data-dir", default=None, help="defaults to data/midv2020_tamper")
    args = parser.parse_args()
    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        preprocessing=args.preprocessing,
        checkpoint_path=args.checkpoint,
        data_dir=Path(args.data_dir) if args.data_dir else None,
    )


if __name__ == "__main__":
    main()

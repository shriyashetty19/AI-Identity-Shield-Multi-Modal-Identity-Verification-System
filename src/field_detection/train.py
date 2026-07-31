import argparse

import torch
from torch.utils.data import DataLoader

from src.common.config import MIDV2020_DIR, MODELS_DIR
from src.field_detection.dataset import FieldDetectionDataset, collate_fn
from src.field_detection.model import build_model


def train(epochs: int = 10, batch_size: int = 4, lr: float = 0.005, checkpoint_path=None) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    train_ds = FieldDetectionDataset(MIDV2020_DIR / "train")
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    model = build_model(pretrained=True).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=5e-4)

    checkpoint_path = checkpoint_path or MODELS_DIR / "field_detector.pt"

    print(f"{len(train_loader)} batches/epoch", flush=True)

    model.train()
    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        for step, (images, targets) in enumerate(train_loader, start=1):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if step % 20 == 0:
                print(f"  epoch {epoch}/{epochs} step {step}/{len(train_loader)}", flush=True)

        print(f"epoch {epoch}/{epochs}  train_loss={running_loss / len(train_loader):.4f}", flush=True)

    torch.save(model.state_dict(), checkpoint_path)
    print(f"saved checkpoint to {checkpoint_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()
    train(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, checkpoint_path=args.checkpoint)


if __name__ == "__main__":
    main()

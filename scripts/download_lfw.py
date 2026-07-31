"""Download Labeled Faces in the Wild (LFW) into data/lfw/.

Dataset: LFW, https://vis-www.cs.umass.edu/lfw/
License: free for research use (see homepage for details).
Used in Phase 2 to validate the pretrained ArcFace/FaceNet face-matching model.
"""
from pathlib import Path

from sklearn.datasets import fetch_lfw_pairs

ROOT = Path(__file__).resolve().parent.parent
DEST = ROOT / "data" / "lfw"


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    if (DEST / "lfw_home").exists():
        print(f"Dataset already present at {DEST}, skipping download.")
        return

    # sklearn caches under <data_home>/lfw_home, i.e. data/lfw/lfw_home/.
    fetch_lfw_pairs(subset="train", data_home=str(DEST), download_if_missing=True)
    fetch_lfw_pairs(subset="test", data_home=str(DEST), download_if_missing=True)
    print(f"Downloaded LFW to {DEST}")


if __name__ == "__main__":
    main()

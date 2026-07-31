"""Download the MIDV-2020 (Roboflow Universe) dataset into data/midv2020/.

Dataset: MIDV-2020 MRP, https://universe.roboflow.com/shriya-shetty-peuyp/midv-2020-mrp-s7pj8
License: CC BY 4.0

Requires a free Roboflow API key: https://app.roboflow.com/settings/api
Set it via a .env file (copy .env.example -> .env) or the ROBOFLOW_API_KEY
environment variable -- never hardcode it here.
"""
import os
import shutil
from pathlib import Path

from dotenv import load_dotenv
from roboflow import Roboflow

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
DEST = ROOT / "data" / "midv2020"


def main() -> None:
    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        raise SystemExit(
            "ROBOFLOW_API_KEY is not set. Copy .env.example to .env and fill in "
            "your free key from https://app.roboflow.com/settings/api"
        )

    if (DEST / "train").exists():
        print(f"Dataset already present at {DEST}, skipping download.")
        return

    rf = Roboflow(api_key=api_key)
    project = rf.workspace("shriya-shetty-peuyp").project("midv-2020-mrp-s7pj8")
    version = project.version(1)
    downloaded = Path(version.download("coco").location)

    DEST.mkdir(parents=True, exist_ok=True)
    for item in downloaded.iterdir():
        shutil.move(str(item), str(DEST / item.name))
    downloaded.rmdir()
    print(f"Downloaded MIDV-2020 to {DEST}")


if __name__ == "__main__":
    main()

"""Download the FUNSD dataset into data/funsd/.

Dataset: FUNSD (Form Understanding in Noisy Scanned Documents)
Homepage: https://guillaumejaume.github.io/FUNSD/
License: research/non-commercial use (see homepage) - used here for the OCR
field-extraction module as a general form-understanding reference set.
"""
import shutil
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

ROOT = Path(__file__).resolve().parent.parent
DEST = ROOT / "data" / "funsd"
ZIP_URL = "https://guillaumejaume.github.io/FUNSD/dataset.zip"


def main() -> None:
    if (DEST / "training_data").exists():
        print(f"Dataset already present at {DEST}, skipping download.")
        return

    DEST.mkdir(parents=True, exist_ok=True)
    zip_path = DEST / "_funsd.zip"
    print(f"Downloading FUNSD from {ZIP_URL} ...")
    urlretrieve(ZIP_URL, zip_path)

    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(DEST)
    zip_path.unlink()

    # The archive nests everything under dataset/{training_data,testing_data}
    nested = DEST / "dataset"
    if nested.exists():
        for item in nested.iterdir():
            shutil.move(str(item), str(DEST / item.name))
        nested.rmdir()

    print(f"Downloaded FUNSD to {DEST}")


if __name__ == "__main__":
    main()

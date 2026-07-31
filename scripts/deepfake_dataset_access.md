# Phase 2: FaceForensics++ / Celeb-DF access

Both datasets are free but gated behind a short academic-use request form -
there is no direct download URL, so this can't be scripted like the Phase 1
datasets. Request access before starting Phase 2:

## FaceForensics++
1. Read the terms and fill out the request form linked from the official repo:
   https://github.com/ondyari/FaceForensics
2. You'll receive a download script (`download_ffpp.py` or similar) by email/
   response, tied to your approved request - run it with your own credentials.
3. Do NOT commit that script or its credentials to this repo; keep it outside
   version control (e.g. in `scripts/private/`, already covered by `.gitignore`
   via the `data/` and secrets rules) and only place the extracted frames
   under `data/faceforensics/`.

## Celeb-DF
1. Request access via the official repo: https://github.com/yuezunli/celeb-deepfakeforensics
2. Follow the same pattern: place extracted frames under `data/celebdf/`,
   keep any provided download credentials/scripts out of the repo.

Once you have either dataset locally, the Phase 2 deepfake-detection notebook
will load frames directly from `data/faceforensics/` or `data/celebdf/` - no
code changes needed beyond pointing `DATA_DIR` at whichever one you obtained.

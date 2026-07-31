"""Classical (non-ML) copy-move forgery detector.

`tampering.py`'s copy_paste technique relocates a patch *within the same
image* - there's no cross-source signal (different compression history,
different resize/resampling) for a classifier to learn from, which is why
the ResNet-50 forgery classifier never picked it up (see the forgery-
detection hardening notes - copy_paste was pulled out of its training set
for exactly this reason). Copy-move is a much older, already-solved forensic
problem instead: find keypoints whose local texture matches another region
of the same image under a consistent spatial offset. That's what real
copy-move detectors (SIFT/ORB self-matching, block-matching) do, and it
needs no training data at all.
"""
from collections import Counter

import cv2
import numpy as np
from PIL import Image

MIN_SPATIAL_DISTANCE = 24  # px - closer than this is just natural local self-similarity
MAX_DESCRIPTOR_DISTANCE = 40  # ORB/Hamming distance threshold for "same texture"
OFFSET_BUCKET_SIZE = 6  # px - groups near-identical offsets together
MIN_CLUSTER_SIZE = 6  # matches sharing an offset needed to call it a detection


def detect_copy_move(image: Image.Image) -> dict:
    gray = np.array(image.convert("L"))
    orb = cv2.ORB_create(nfeatures=1500)
    keypoints, descriptors = orb.detectAndCompute(gray, None)

    if descriptors is None or len(keypoints) < 2:
        return {"detected": False, "match_count": 0, "offset": None}

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    k = min(3, len(keypoints))
    all_matches = bf.knnMatch(descriptors, descriptors, k=k)

    offsets = []
    for matches in all_matches:
        for m in matches[1:]:  # index 0 is always the trivial self-match (distance 0)
            # self-matching finds every (i, j) pair from both sides (i->j and
            # j->i); keep only one direction so a duplicated region isn't
            # split across two opposite-signed offset buckets.
            if m.trainIdx <= m.queryIdx:
                continue
            if m.distance > MAX_DESCRIPTOR_DISTANCE:
                continue
            p1 = np.array(keypoints[m.queryIdx].pt)
            p2 = np.array(keypoints[m.trainIdx].pt)
            if np.linalg.norm(p1 - p2) < MIN_SPATIAL_DISTANCE:
                continue
            dx, dy = p2 - p1
            offsets.append((round(dx / OFFSET_BUCKET_SIZE), round(dy / OFFSET_BUCKET_SIZE)))

    if not offsets:
        return {"detected": False, "match_count": 0, "offset": None}

    bucket, count = Counter(offsets).most_common(1)[0]
    detected = count >= MIN_CLUSTER_SIZE
    offset = (bucket[0] * OFFSET_BUCKET_SIZE, bucket[1] * OFFSET_BUCKET_SIZE) if detected else None
    return {"detected": detected, "match_count": count, "offset": offset}

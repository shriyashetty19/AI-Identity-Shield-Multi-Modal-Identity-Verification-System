import torch
from torch import nn
from torchvision.models.detection import (
    FasterRCNN_MobileNet_V3_Large_FPN_Weights,
    fasterrcnn_mobilenet_v3_large_fpn,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

LABELS = [
    "background",
    "primary_identifier",
    "date_of_birth",
    "document_number",
    "face_image",
    "date_of_expiry",
    "date_of_issue",
    "personal_number",
    "secondary_identifier",
    "place_of_birth",
]

# category name in the MIDV-2020 annotations -> label index this model
# predicts. Covers every category tampering.py's splice/photo_swap can
# target (see TEXT_FIELDS + face_image there), not just the 3 fields OCR
# needs - src.forgery_detection.region_inference uses the same detector to
# localize every tamper-prone region before classifying each crop.
CATEGORY_TO_LABEL = {
    "primary_identifier": 1,
    "date_of_birth": 2,
    "document_number": 3,
    "face_image": 4,
    "date_of_expiry": 5,
    "date_of_issue": 6,
    "personal_number": 7,
    "secondary_identifier": 8,
    "place_of_birth": 9,
}


def build_model(pretrained: bool = True) -> nn.Module:
    # MobileNetV3 instead of the heavier ResNet-50-FPN-v2 backbone: benchmarked
    # on this GPU at ~15x faster per batch (0.3s vs 4.9s) with no accuracy
    # tradeoff that matters here - the 3 target fields are large, visually
    # distinctive regions on one consistent document template, not a subtle
    # detection problem that needs a heavyweight backbone.
    weights = FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT if pretrained else None
    model = fasterrcnn_mobilenet_v3_large_fpn(weights=weights)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, len(LABELS))
    return model


def load_model(checkpoint_path, device: str = "cpu") -> nn.Module:
    model = build_model(pretrained=False)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model

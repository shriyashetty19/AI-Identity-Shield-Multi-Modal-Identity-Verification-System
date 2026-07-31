import json

from src.common.coco_utils import largest_box, load_coco_split

SAMPLE_COCO = {
    "categories": [
        {"id": 0, "name": "document"},
        {"id": 1, "name": "face_image"},
        {"id": 2, "name": "date_of_birth"},
    ],
    "images": [
        {"id": 0, "file_name": "doc_0.jpg", "width": 512, "height": 512},
        {"id": 1, "file_name": "doc_1.jpg", "width": 512, "height": 512},
    ],
    "annotations": [
        {"id": 0, "image_id": 0, "category_id": 1, "bbox": [10, 10, 50, 50]},
        {"id": 1, "image_id": 0, "category_id": 2, "bbox": [100, 100, 80, 20]},
        # image 0 has two "document" boxes; largest_box should pick the bigger one
        {"id": 2, "image_id": 0, "category_id": 0, "bbox": [0, 0, 20, 20]},
        {"id": 3, "image_id": 0, "category_id": 0, "bbox": [0, 0, 200, 200]},
    ],
}


def test_load_coco_split_groups_fields_by_image(tmp_path):
    (tmp_path / "_annotations.coco.json").write_text(json.dumps(SAMPLE_COCO))

    records = load_coco_split(tmp_path)

    assert len(records) == 2
    doc0 = next(r for r in records if r.file_name == "doc_0.jpg")
    doc1 = next(r for r in records if r.file_name == "doc_1.jpg")

    assert "face_image" in doc0.fields
    assert "date_of_birth" in doc0.fields
    assert doc1.fields == {}  # no annotations for image 1


def test_largest_box_picks_biggest_area():
    boxes = [[0, 0, 20, 20], [0, 0, 200, 200], [5, 5, 10, 10]]
    assert largest_box(boxes) == [0, 0, 200, 200]

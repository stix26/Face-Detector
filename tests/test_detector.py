from pathlib import Path
import cv2
from face_detector import detect_faces, annotate_image

TEST_IMAGE = Path(__file__).resolve().parent.parent / "photo.jpg"


def test_detect_faces():
    img, faces = detect_faces(TEST_IMAGE)
    assert img is not None
    assert len(faces) > 0


def test_annotate_image(tmp_path):
    img, faces = detect_faces(TEST_IMAGE)
    annotated = annotate_image(img.copy(), faces)
    output = tmp_path / "out.jpg"
    cv2.imwrite(str(output), annotated)
    assert output.exists() and output.stat().st_size > 0

# Face Detector

This project demonstrates simple face detection using OpenCV's Haar cascade classifier.

## Requirements
- Python 3.8+
- `opencv-python-headless`
- `pytest` (for running tests)

Install dependencies:
```bash
pip install -e .[test]
```

## Usage
Run face detection on an image:
```bash
python face_detector.py path/to/image.jpg --output annotated.jpg
```
Use `--display` to show the result in a window (requires GUI support).

## Testing
Unit tests are provided with `pytest`:
```bash
pytest
```

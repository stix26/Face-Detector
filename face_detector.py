import argparse
from pathlib import Path

import cv2

CASCADE_PATH = Path(__file__).with_name("haarcascade_frontalface_default.xml")

def detect_faces(image_path: Path):
    """Load an image and return it along with detected face rectangles."""
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Unable to read image {image_path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier(str(CASCADE_PATH))
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    return img, faces

def annotate_image(img, faces):
    for x, y, w, h in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return img

def main(argv=None):
    parser = argparse.ArgumentParser(description="Detect faces in an image")
    parser.add_argument("image", type=Path, help="Path to input image")
    parser.add_argument("--output", type=Path, default=Path("output.jpg"), help="Output annotated image path")
    parser.add_argument("--display", action="store_true", help="Display the annotated image")
    args = parser.parse_args(argv)

    img, faces = detect_faces(args.image)
    annotated = annotate_image(img.copy(), faces)
    cv2.imwrite(str(args.output), annotated)
    print(f"Found {len(faces)} face(s). Saved result to {args.output}")

    if args.display:
        cv2.imshow("Faces", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

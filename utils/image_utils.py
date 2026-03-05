import cv2
import base64

MAX_WIDTH = 854
MAX_HEIGHT = 480
JPEG_QUALITY = 78


def preprocess_frame(frame):
    """
    Downscales to a lightweight size (preserving aspect ratio),
    converts to JPEG, and base64 encodes it.
    """
    height, width = frame.shape[:2]
    scale = min(MAX_WIDTH / float(width), MAX_HEIGHT / float(height), 1.0)
    if scale < 1.0:
        new_w = max(1, int(round(width * scale)))
        new_h = max(1, int(round(height * scale)))
        frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Convert to JPEG at reduced quality for smaller payloads.
    success, buffer = cv2.imencode(
        '.jpg',
        frame,
        [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY],
    )
    if not success:
        raise ValueError("Failed to encode frame as JPEG.")
    # Return as base64 string
    return base64.b64encode(buffer).decode('utf-8')

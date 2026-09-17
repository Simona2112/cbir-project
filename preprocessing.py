"""
preprocessing.py

Functions for preparing images before feature extraction.
"""

import cv2
import numpy as np

from config import IMAGE_SIZE, MIN_CONTRAST_STD


def load_image(path):
    """Load an image from disk."""

    try:
        raw_bytes = np.fromfile(
            path,
            dtype=np.uint8
        )

        if raw_bytes.size == 0:
            return None

        return cv2.imdecode(
            raw_bytes,
            cv2.IMREAD_COLOR
        )

    except Exception:
        return None


def resize_with_padding(image, size=IMAGE_SIZE):
    """Resize an image without changing its aspect ratio."""

    height, width = image.shape[:2]

    scale = size / float(
        max(height, width)
    )

    new_width = max(
        1,
        int(round(width * scale))
    )

    new_height = max(
        1,
        int(round(height * scale))
    )

    resized = cv2.resize(
        image,
        (new_width, new_height),
        interpolation=cv2.INTER_AREA
    )

    pad_top = (
        size - new_height
    ) // 2

    pad_bottom = (
        size - new_height
        - pad_top
    )

    pad_left = (
        size - new_width
    ) // 2

    pad_right = (
        size - new_width
        - pad_left
    )

    return cv2.copyMakeBorder(
        resized,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        borderType=cv2.BORDER_CONSTANT,
        value=(0, 0, 0)
    )


def to_grayscale(image):
    """Convert a BGR image to grayscale."""

    return cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )


def normalize_lighting(gray):
    """Improve local contrast using CLAHE."""

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    return clahe.apply(gray)


def has_enough_contrast(gray):
    """Check whether the image has enough contrast."""

    return (
        float(gray.std())
        >= MIN_CONTRAST_STD
    )


def preprocess(image):
    """Run the complete preprocessing pipeline."""

    padded = resize_with_padding(
        image,
        IMAGE_SIZE
    )

    gray = to_grayscale(padded)

    # Check contrast before applying CLAHE
    usable = has_enough_contrast(gray)

    gray = normalize_lighting(gray)

    return gray, usable


def load_and_preprocess(path):
    """Load and preprocess an image."""

    image = load_image(path)

    if image is None:
        return None, False

    return preprocess(image)
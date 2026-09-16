"""
preprocessing.py
----------------
Every image in the dataset has a different size and a different shape
(some are wide, some are tall). Before we can compare two images we must
bring them all to the SAME size, otherwise the feature vectors would not
be comparable.

Steps:
    read image
        -> resize keeping the aspect ratio
        -> add padding so every image becomes 256x256
        -> convert to grayscale
        -> normalise the lighting with CLAHE
"""

import os
import cv2
import numpy as np

from config import IMAGE_SIZE, MIN_CONTRAST_STD


def load_image(path):
    """
    Read an image from disk with OpenCV.

    We use np.fromfile + cv2.imdecode instead of cv2.imread because
    cv2.imread fails on Windows when the path contains non-English letters.

    Returns the image as a BGR NumPy array, or None if the file is not a
    valid image (broken file, wrong format, ...). The caller must check
    for None so that one bad file does not crash the whole program.
    """
    try:
        # Read the raw bytes of the file into a NumPy array of type uint8
        raw_bytes = np.fromfile(path, dtype=np.uint8)
        if raw_bytes.size == 0:
            return None

        # Decode those bytes into a real image (BGR, 3 channels)
        image = cv2.imdecode(raw_bytes, cv2.IMREAD_COLOR)
        return image
    except Exception:
        # Any problem at all -> we simply say "this image could not be loaded"
        return None


def resize_with_padding(image, size=IMAGE_SIZE):
    """
    Resize the image to size x size WITHOUT stretching it.

    If we simply called cv2.resize(image, (256, 256)) a tall image would be
    squeezed and a wide image would be stretched. That would change the shape
    of the object, and our shape descriptor would be wrong.

    Instead we:
        1. scale the image so that its LONGER side becomes 256 pixels
        2. put the scaled image in the middle of a black 256x256 canvas

    This is called "letterboxing" and it keeps the original aspect ratio.
    """
    height, width = image.shape[:2]

    # Scale factor = how much we must shrink/grow the image so the longer
    # side becomes exactly 'size' pixels.
    scale = size / max(height, width)

    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))

    # INTER_AREA gives the best quality when we make an image smaller
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

    # How many black pixels we must add on each side to reach size x size
    pad_top = (size - new_height) // 2
    pad_bottom = size - new_height - pad_top
    pad_left = (size - new_width) // 2
    pad_right = size - new_width - pad_left

    # cv2.copyMakeBorder adds a border around the image.
    # BORDER_CONSTANT with value 0 means "fill it with black".
    padded = cv2.copyMakeBorder(
        resized,
        pad_top, pad_bottom, pad_left, pad_right,
        borderType=cv2.BORDER_CONSTANT,
        value=(0, 0, 0)
    )
    return padded


def to_grayscale(image):
    """
    Convert a BGR colour image to a single channel grayscale image.

    IMPORTANT for this project: we are NOT allowed to use colour information,
    so every descriptor works on the grayscale image only. Converting to
    grayscale here guarantees that no colour can influence the results.
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def normalize_lighting(gray):
    """
    Make the image independent of the lighting of the photo.

    Photographs of the same object can be taken in very different light. A
    picture taken in the shade and the same object in bright sun would give
    different LBP histograms and different edges, even though the object is
    the same. This step removes that difference.

    We use CLAHE = Contrast Limited Adaptive Histogram Equalization.

    Normal histogram equalization stretches the histogram of the WHOLE image,
    which often over-amplifies the noise. CLAHE instead:
        - cuts the image into small tiles (here 8x8 tiles)
        - equalizes the histogram inside every tile separately
        - limits ("clips") the contrast so noise is not amplified too much
        - joins the tiles back together smoothly

    The result is that a dark photo and a bright photo of the same object
    give almost the same grayscale image, and therefore almost the same
    features. This is a classical Digital Image Processing operation.
    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def has_enough_contrast(gray):
    """
    Check whether an image contains enough information to be used.

    We measure the STANDARD DEVIATION of the grayscale values. The standard
    deviation tells us how much the pixel values differ from the average
    brightness:

        big value -> bright and dark areas, a lot of information
        small value -> the whole image has almost the same value

    Some images in this dataset were darkened so strongly that all their
    pixels lie between 0 and 30 - to the eye they are simply black
    rectangles. If we send such an image through CLAHE, the contrast
    stretching only amplifies the JPEG noise, and the noise then matches
    everything. In a first test one of those black images appeared in the
    Top 10 of EVERY query, which is of course wrong.

    So we check the contrast BEFORE CLAHE and skip images that are too flat.
    This must be done before CLAHE, because after CLAHE even a black image
    looks like it has contrast.
    """
    return float(gray.std()) >= MIN_CONTRAST_STD


def preprocess(image):
    """
    The complete preprocessing pipeline used everywhere in the project.

    Input : BGR image of any size
    Output: (gray, usable)
            gray   = grayscale image of exactly IMAGE_SIZE x IMAGE_SIZE pixels
                     with normalised lighting
            usable = False if the original image was almost black / flat

        resize + padding  ->  grayscale  ->  contrast check  ->  CLAHE
    """
    padded = resize_with_padding(image, IMAGE_SIZE)
    gray = to_grayscale(padded)

    # remember whether the image had contrast BEFORE we stretch it
    usable = has_enough_contrast(gray)

    gray = normalize_lighting(gray)
    return gray, usable


def load_and_preprocess(path):
    """
    Convenience function: load a file from disk and preprocess it.

    Returns (gray, usable), or (None, False) if the file could not be loaded.
    """
    image = load_image(path)
    if image is None:
        return None, False
    return preprocess(image)

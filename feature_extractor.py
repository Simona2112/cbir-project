"""
feature_extractor.py

Extracts texture, shape and local descriptors from a grayscale image.
"""

import cv2
import numpy as np

from config import (
    ORB_N_FEATURES,
    SHAPE_MIN_CONTOUR_RATIO,
    SHAPE_MAX_CONTOUR_RATIO,
    LBP_GRID,
    LBP_BINS,
    LBP_SCALES,
    USE_ROI,
    ROI_MARGIN,
    ROI_MIN_FRACTION,
    ROI_MAX_FRACTION,
    ROI_MIN_SIDE,
    IMAGE_SIZE,
    SHAPE_DESCRIPTOR,
    HOG_CELL_SIZE,
    HOG_BINS,
    LOCAL_DESCRIPTOR,
    SIFT_N_FEATURES
)


# Create detectors once and reuse them
orb_detector = cv2.ORB_create(nfeatures=ORB_N_FEATURES)
sift_detector = cv2.SIFT_create(nfeatures=SIFT_N_FEATURES)


# ============================================================
# TEXTURE - LBP
# ============================================================

def compute_lbp_image(gray, radius=1):
    """Calculate an LBP code for every pixel."""

    height, width = gray.shape

    padded = cv2.copyMakeBorder(
        gray,
        radius,
        radius,
        radius,
        radius,
        cv2.BORDER_REPLICATE
    ).astype(np.int16)

    centre = padded[
        radius:radius + height,
        radius:radius + width
    ]

    offsets = [
        (-radius, -radius),
        (-radius, 0),
        (-radius, radius),
        (0, radius),
        (radius, radius),
        (radius, 0),
        (radius, -radius),
        (0, -radius)
    ]

    lbp = np.zeros(
        centre.shape,
        dtype=np.uint8
    )

    for bit_index, (row_offset, column_offset) in enumerate(offsets):

        neighbour = padded[
            radius + row_offset:
            radius + row_offset + height,

            radius + column_offset:
            radius + column_offset + width
        ]

        bit = (
            neighbour >= centre
        ).astype(np.uint8)

        lbp += bit * (1 << bit_index)

    return lbp


def extract_texture_features(gray):
    """Create the spatial LBP descriptor."""

    histograms = []

    for radius in LBP_SCALES:

        lbp = compute_lbp_image(
            gray,
            radius=radius
        )

        height, width = lbp.shape

        block_height = height // LBP_GRID
        block_width = width // LBP_GRID

        for row in range(LBP_GRID):
            for column in range(LBP_GRID):

                y_start = row * block_height
                x_start = column * block_width

                block = lbp[
                    y_start:y_start + block_height,
                    x_start:x_start + block_width
                ]

                histogram = np.bincount(
                    block.ravel(),
                    minlength=LBP_BINS
                ).astype(np.float32)

                total = histogram.sum()

                if total > 0:
                    histogram /= total

                histograms.append(histogram)

    return np.concatenate(histograms)


# ============================================================
# OLD SHAPE DESCRIPTOR - HU MOMENTS
# ============================================================

def choose_main_contour(contours, image_area):
    """Choose the most likely main object contour."""

    if len(contours) == 0:
        return None

    image_side = int(
        round(np.sqrt(image_area))
    )

    good_contours = []

    for contour in contours:

        area = cv2.contourArea(contour)

        if area < SHAPE_MIN_CONTOUR_RATIO * image_area:
            continue

        if area > SHAPE_MAX_CONTOUR_RATIO * image_area:
            continue

        x, y, width, height = cv2.boundingRect(contour)

        if (
            x <= 1
            and y <= 1
            and width >= image_side - 2
            and height >= image_side - 2
        ):
            continue

        good_contours.append(contour)

    if len(good_contours) == 0:
        return max(
            contours,
            key=cv2.contourArea
        )

    return max(
        good_contours,
        key=cv2.contourArea
    )


def extract_shape_features(gray):
    """Extract the older Hu-moments shape descriptor."""

    height, width = gray.shape
    image_area = float(height * width)

    blurred = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    edges = cv2.Canny(
        blurred,
        50,
        150
    )

    kernel = np.ones(
        (3, 3),
        dtype=np.uint8
    )

    edges = cv2.dilate(
        edges,
        kernel,
        iterations=1
    )

    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) == 0:
        return np.zeros(
            12,
            dtype=np.float32
        )

    main_contour = choose_main_contour(
        contours,
        image_area
    )

    if main_contour is None:
        return np.zeros(
            12,
            dtype=np.float32
        )

    area = cv2.contourArea(main_contour)
    perimeter = cv2.arcLength(
        main_contour,
        closed=True
    )

    if area <= 0 or perimeter <= 0:
        return np.zeros(
            12,
            dtype=np.float32
        )

    moments = cv2.moments(main_contour)

    hu_moments = cv2.HuMoments(
        moments
    ).flatten()

    hu_log = np.zeros(
        7,
        dtype=np.float32
    )

    for i in range(7):
        value = hu_moments[i]

        hu_log[i] = (
            -np.sign(value)
            * np.log10(abs(value) + 1e-30)
        )

    hu_log = np.clip(
        hu_log / 30.0,
        -1.0,
        1.0
    )

    area_ratio = area / image_area

    perimeter_ratio = min(
        perimeter / (16.0 * width),
        1.0
    )

    circularity = min(
        (4.0 * np.pi * area)
        / (perimeter * perimeter),
        1.0
    )

    x, y, w, h = cv2.boundingRect(
        main_contour
    )

    aspect_ratio = (
        float(w) / float(h)
        if h > 0
        else 0.0
    )

    aspect_ratio = min(
        aspect_ratio / 3.0,
        1.0
    )

    extent = (
        area / float(w * h)
        if (w * h) > 0
        else 0.0
    )

    extra = np.array(
        [
            area_ratio,
            perimeter_ratio,
            circularity,
            aspect_ratio,
            extent
        ],
        dtype=np.float32
    )

    return np.concatenate(
        [hu_log, extra]
    ).astype(np.float32)


# ============================================================
# SHAPE - HOG
# ============================================================

def extract_hog_features(gray):
    """Extract the HOG shape descriptor."""

    image = gray.astype(np.float32)

    gradient_x = cv2.Sobel(
        image,
        cv2.CV_32F,
        1,
        0,
        ksize=3
    )

    gradient_y = cv2.Sobel(
        image,
        cv2.CV_32F,
        0,
        1,
        ksize=3
    )

    magnitude = cv2.magnitude(
        gradient_x,
        gradient_y
    )

    orientation = np.rad2deg(
        np.arctan2(
            gradient_y,
            gradient_x
        )
    ) % 180.0

    degrees_per_bin = (
        180.0 / HOG_BINS
    )

    bin_index = (
        orientation / degrees_per_bin
    ).astype(np.int32)

    bin_index = np.minimum(
        bin_index,
        HOG_BINS - 1
    )

    cells_y = (
        gray.shape[0]
        // HOG_CELL_SIZE
    )

    cells_x = (
        gray.shape[1]
        // HOG_CELL_SIZE
    )

    histograms = []

    for row in range(cells_y):
        for column in range(cells_x):

            y_start = row * HOG_CELL_SIZE
            x_start = column * HOG_CELL_SIZE

            cell_bins = bin_index[
                y_start:y_start + HOG_CELL_SIZE,
                x_start:x_start + HOG_CELL_SIZE
            ].ravel()

            cell_magnitude = magnitude[
                y_start:y_start + HOG_CELL_SIZE,
                x_start:x_start + HOG_CELL_SIZE
            ].ravel()

            histogram = np.bincount(
                cell_bins,
                weights=cell_magnitude,
                minlength=HOG_BINS
            )

            histograms.append(histogram)

    vector = np.concatenate(
        histograms
    ).astype(np.float32)

    norm = float(
        np.linalg.norm(vector)
    )

    if norm > 0:
        vector /= norm + 1e-10

    return vector.astype(np.float32)


# ============================================================
# LOCAL FEATURES
# ============================================================

def extract_orb_features(gray):
    """Extract ORB keypoints and descriptors."""

    keypoints, descriptors = (
        orb_detector.detectAndCompute(
            gray,
            None
        )
    )

    if descriptors is None:
        return 0, None

    return len(keypoints), descriptors


def extract_sift_features(gray):
    """Extract SIFT keypoints and descriptors."""

    keypoints, descriptors = (
        sift_detector.detectAndCompute(
            gray,
            None
        )
    )

    if descriptors is None:
        return 0, None

    return len(keypoints), descriptors


# ============================================================
# ROI
# ============================================================

def extract_main_roi(gray):
    """Extract a possible main-object region."""

    height, width = gray.shape
    image_area = float(height * width)

    blurred = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    edges = cv2.Canny(
        blurred,
        50,
        150
    )

    kernel = np.ones(
        (3, 3),
        dtype=np.uint8
    )

    edges = cv2.dilate(
        edges,
        kernel,
        iterations=1
    )

    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    main_contour = choose_main_contour(
        contours,
        image_area
    )

    if main_contour is None:
        return gray

    x, y, box_width, box_height = (
        cv2.boundingRect(main_contour)
    )

    margin_x = int(
        round(box_width * ROI_MARGIN)
    )

    margin_y = int(
        round(box_height * ROI_MARGIN)
    )

    x_start = max(
        0,
        x - margin_x
    )

    y_start = max(
        0,
        y - margin_y
    )

    x_end = min(
        width,
        x + box_width + margin_x
    )

    y_end = min(
        height,
        y + box_height + margin_y
    )

    roi_width = x_end - x_start
    roi_height = y_end - y_start

    if (
        roi_width < ROI_MIN_SIDE
        or roi_height < ROI_MIN_SIDE
    ):
        return gray

    covered = (
        roi_width * roi_height
    ) / image_area

    if (
        covered < ROI_MIN_FRACTION
        or covered > ROI_MAX_FRACTION
    ):
        return gray

    roi = gray[
        y_start:y_end,
        x_start:x_end
    ]

    scale = (
        IMAGE_SIZE
        / float(max(roi_height, roi_width))
    )

    new_width = max(
        1,
        int(round(roi_width * scale))
    )

    new_height = max(
        1,
        int(round(roi_height * scale))
    )

    resized = cv2.resize(
        roi,
        (new_width, new_height),
        interpolation=cv2.INTER_AREA
    )

    pad_top = (
        IMAGE_SIZE - new_height
    ) // 2

    pad_bottom = (
        IMAGE_SIZE
        - new_height
        - pad_top
    )

    pad_left = (
        IMAGE_SIZE - new_width
    ) // 2

    pad_right = (
        IMAGE_SIZE
        - new_width
        - pad_left
    )

    return cv2.copyMakeBorder(
        resized,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        borderType=cv2.BORDER_CONSTANT,
        value=0
    )


# ============================================================
# ALL FEATURES
# ============================================================

def extract_all_features(gray):
    """Extract texture, shape and local features."""

    if USE_ROI:
        texture_source = extract_main_roi(gray)
    else:
        texture_source = gray

    if LOCAL_DESCRIPTOR == "sift":
        keypoint_count, local_descriptors = (
            extract_sift_features(texture_source)
        )
    else:
        keypoint_count, local_descriptors = (
            extract_orb_features(texture_source)
        )

    if SHAPE_DESCRIPTOR == "hog":
        shape_vector = extract_hog_features(gray)
    else:
        shape_vector = extract_shape_features(gray)

    return {
        "texture": extract_texture_features(
            texture_source
        ),
        "shape": shape_vector,
        "orb": local_descriptors,
        "keypoints": keypoint_count
    }


# ============================================================
# WEIGHTING HELPERS
# ============================================================

def orb_keypoint_info(gray):
    """Return ORB keypoint count and average response."""

    keypoints = orb_detector.detect(
        gray,
        None
    )

    if keypoints is None or len(keypoints) == 0:
        return 0, 0.0

    average_response = float(
        np.mean(
            [
                point.response
                for point in keypoints
            ]
        )
    )

    return (
        len(keypoints),
        average_response
    )
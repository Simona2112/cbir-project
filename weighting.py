"""
weighting.py

Calculates automatic descriptor weights from the query image.
"""

import cv2
import numpy as np

from config import (
    ORB_N_FEATURES,
    LAPLACIAN_STD_REFERENCE,
    EDGE_DENSITY_REFERENCE,
    CONTOUR_AREA_REFERENCE,
    ORB_RESPONSE_REFERENCE,
    LOCAL_TRUST,
    MINIMUM_IMPORTANCE,
    SHAPE_AREA_PART,
    SHAPE_SOLIDITY_PART,
    SHAPE_EDGE_PART
)

from feature_extractor import (
    orb_keypoint_info,
    choose_main_contour
)


# ============================================================
# TEXTURE IMPORTANCE
# ============================================================

def texture_importance(gray):
    """Estimate how much fine texture the image contains."""

    laplacian = cv2.Laplacian(
        gray,
        cv2.CV_64F
    )

    detail = float(
        laplacian.std()
    )

    return min(
        detail / LAPLACIAN_STD_REFERENCE,
        1.0
    )


# ============================================================
# SHAPE IMPORTANCE
# ============================================================

def shape_importance(gray):
    """Estimate how strong the main object shape is."""

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

    total_pixels = float(
        edges.shape[0] * edges.shape[1]
    )

    # Small contribution from edge density
    edge_density = (
        np.count_nonzero(edges)
        / total_pixels
    )

    edge_score = min(
        edge_density / EDGE_DENSITY_REFERENCE,
        1.0
    )

    kernel = np.ones(
        (3, 3),
        dtype=np.uint8
    )

    closed_edges = cv2.dilate(
        edges,
        kernel,
        iterations=1
    )

    contours, _ = cv2.findContours(
        closed_edges,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    main_contour = choose_main_contour(
        contours,
        total_pixels
    )

    if main_contour is None:
        return (
            SHAPE_EDGE_PART
            * edge_score
        )

    contour_area = cv2.contourArea(
        main_contour
    )

    if contour_area <= 0:
        return (
            SHAPE_EDGE_PART
            * edge_score
        )

    # Main object size
    area_ratio = (
        contour_area / total_pixels
    )

    area_score = min(
        area_ratio / CONTOUR_AREA_REFERENCE,
        1.0
    )

    # Solidity of the main contour
    hull = cv2.convexHull(
        main_contour
    )

    hull_area = cv2.contourArea(
        hull
    )

    if hull_area > 0:
        solidity = (
            contour_area / hull_area
        )
    else:
        solidity = 0.0

    solidity = float(
        np.clip(
            solidity,
            0.0,
            1.0
        )
    )

    return (
        SHAPE_AREA_PART * area_score
        + SHAPE_SOLIDITY_PART * solidity
        + SHAPE_EDGE_PART * edge_score
    )


# ============================================================
# LOCAL FEATURE IMPORTANCE
# ============================================================

def local_importance(gray):
    """Estimate the importance of ORB local features."""

    keypoint_count, average_response = (
        orb_keypoint_info(gray)
    )

    count_score = min(
        keypoint_count
        / float(ORB_N_FEATURES),
        1.0
    )

    strength_score = min(
        average_response
        / ORB_RESPONSE_REFERENCE,
        1.0
    )

    raw_importance = (
        0.5 * count_score
        + 0.5 * strength_score
    )

    return (
        raw_importance
        * LOCAL_TRUST
    )


# ============================================================
# FINAL WEIGHTS
# ============================================================

def compute_weights(gray, return_raw=False):
    """Calculate and normalize the three descriptor weights."""

    raw_texture = texture_importance(
        gray
    )

    raw_shape = shape_importance(
        gray
    )

    raw_local = local_importance(
        gray
    )

    texture_value = (
        raw_texture
        + MINIMUM_IMPORTANCE
    )

    shape_value = (
        raw_shape
        + MINIMUM_IMPORTANCE
    )

    local_value = (
        raw_local
        + MINIMUM_IMPORTANCE
    )

    total = (
        texture_value
        + shape_value
        + local_value
    )

    if total <= 0:

        weights = {
            "texture": 1.0 / 3.0,
            "shape": 1.0 / 3.0,
            "local": 1.0 / 3.0
        }

    else:

        weights = {
            "texture": texture_value / total,
            "shape": shape_value / total,
            "local": local_value / total
        }

    if return_raw:

        raw = {
            "texture": raw_texture,
            "shape": raw_shape,
            "local": raw_local
        }

        return weights, raw

    return weights
"""
distances.py

Functions for comparing image descriptors.
Smaller distance means more similar images.
"""

import cv2
import numpy as np

from config import (
    ORB_RATIO_TEST,
    ORB_DESCRIPTOR_BITS,
    ORB_GOOD_MATCH_REFERENCE,
    WORST_DISTANCE,
    SHAPE_REFERENCE_DISTANCE,
    LBP_GRID,
    LBP_SCALES,
    LOCAL_DESCRIPTOR,
    SIFT_RATIO_TEST,
    SIFT_GOOD_MATCH_REFERENCE,
    SIFT_MAX_DISTANCE
)


# ============================================================
# TEXTURE DISTANCE - LBP
# ============================================================

# Number of LBP histogram blocks
LBP_BLOCK_COUNT = LBP_GRID * LBP_GRID * len(LBP_SCALES)


def texture_distance(hist1, hist2):
    """Compare two LBP descriptors using Chi-square distance."""

    if hist1 is None or hist2 is None:
        return WORST_DISTANCE

    numerator = (hist1 - hist2) ** 2
    denominator = hist1 + hist2 + 1e-10

    distance = 0.5 * float(
        np.sum(numerator / denominator)
    )

    # Average the distance over all spatial LBP blocks
    distance = distance / LBP_BLOCK_COUNT

    return float(
        np.clip(distance, 0.0, 1.0)
    )


# ============================================================
# SHAPE DISTANCE - HOG
# ============================================================

def shape_distance(shape1, shape2):
    """Compare two HOG descriptors using Euclidean distance."""

    if shape1 is None or shape2 is None:
        return WORST_DISTANCE

    difference = (
        shape1.astype(np.float64)
        - shape2.astype(np.float64)
    )

    raw_distance = float(
        np.sqrt(
            np.sum(difference * difference)
        )
    )

    distance = raw_distance / SHAPE_REFERENCE_DISTANCE

    return float(
        np.clip(distance, 0.0, 1.0)
    )


# ============================================================
# LOCAL FEATURE DISTANCE
# ============================================================

# ORB uses Hamming distance
orb_matcher = cv2.BFMatcher(
    cv2.NORM_HAMMING,
    crossCheck=False
)

# SIFT uses L2 distance
sift_matcher = cv2.BFMatcher(
    cv2.NORM_L2,
    crossCheck=False
)


def local_distance(descriptors1, descriptors2):
    """Compare local descriptors using descriptor matching."""

    if descriptors1 is None or descriptors2 is None:
        return WORST_DISTANCE

    if len(descriptors1) < 2 or len(descriptors2) < 2:
        return WORST_DISTANCE

    # Select settings depending on the active local descriptor
    if LOCAL_DESCRIPTOR == "sift":
        matcher = sift_matcher
        ratio_test = SIFT_RATIO_TEST
        largest_possible = SIFT_MAX_DISTANCE
        match_reference = SIFT_GOOD_MATCH_REFERENCE

    else:
        matcher = orb_matcher
        ratio_test = ORB_RATIO_TEST
        largest_possible = ORB_DESCRIPTOR_BITS
        match_reference = ORB_GOOD_MATCH_REFERENCE

    try:
        matches = matcher.knnMatch(
            descriptors1,
            descriptors2,
            k=2
        )
    except cv2.error:
        return WORST_DISTANCE

    good_distances = []

    # Lowe's ratio test
    for match_pair in matches:

        if len(match_pair) < 2:
            continue

        best, second_best = match_pair

        if best.distance < ratio_test * second_best.distance:
            good_distances.append(best.distance)

    if len(good_distances) == 0:
        return WORST_DISTANCE

    # Average quality of the good matches
    quality = (
        float(np.mean(good_distances))
        / largest_possible
    )

    # More good matches means higher confidence
    confidence = min(
        len(good_distances) / float(match_reference),
        1.0
    )

    distance = (
        confidence * quality
        + (1.0 - confidence) * WORST_DISTANCE
    )

    return float(
        np.clip(distance, 0.0, 1.0)
    )


# ============================================================
# DISTANCE NORMALIZATION
# ============================================================

def normalise_by_mean(distance_list):
    """Normalize one descriptor's distances using their mean."""

    values = np.asarray(
        distance_list,
        dtype=np.float64
    )

    mean_value = values.mean()

    if mean_value <= 1e-12:
        return values

    return values / mean_value


# ============================================================
# FINAL DISTANCE
# ============================================================

def combine_distances(
        texture_values,
        shape_values,
        local_values,
        weights
):
    """Normalize and combine the three descriptor distances."""

    texture_normalised = normalise_by_mean(
        texture_values
    )

    shape_normalised = normalise_by_mean(
        shape_values
    )

    local_normalised = normalise_by_mean(
        local_values
    )

    final = (
        weights["texture"] * texture_normalised
        + weights["shape"] * shape_normalised
        + weights["local"] * local_normalised
    )

    return (
        final,
        texture_normalised,
        shape_normalised,
        local_normalised
    )
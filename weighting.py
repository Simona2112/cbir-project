"""
weighting.py
------------
AUTOMATIC (DYNAMIC) DESCRIPTOR WEIGHTS.

The three descriptors are not equally useful for every image:

    - a photo full of petals or fur is rich in repetitive texture
      -> TEXTURE should count more
    - a car or an apple has one clear, compact silhouette
      -> SHAPE should count more
    - a photo with many strong, sharp corners
      -> LOCAL FEATURES can count more

So instead of fixed weights the program looks at the QUERY IMAGE and decides
by itself. The user never chooses the weights.

The idea is always the same:
    1. measure ONE simple number that says "how much of this kind of
       information is in the image"
    2. divide it by a REFERENCE VALUE so it becomes a number in [0, 1]
    3. normalise the three results so they add up to 1.0

The reference values live in config.py. They are approximately the 90th
percentile of what we measured on this dataset, so a typical image lands in
the middle of the range and only an extreme image reaches 1.0.
"""

import cv2
import numpy as np

from config import (ORB_N_FEATURES, LAPLACIAN_STD_REFERENCE,
                    EDGE_DENSITY_REFERENCE, CONTOUR_AREA_REFERENCE,
                    ORB_RESPONSE_REFERENCE, LOCAL_TRUST, MINIMUM_IMPORTANCE,
                    SHAPE_AREA_PART, SHAPE_SOLIDITY_PART, SHAPE_EDGE_PART)
from feature_extractor import orb_keypoint_info, choose_main_contour


def texture_importance(gray):
    """
    How much TEXTURE / fine detail does the image contain?

    We use the LAPLACIAN. The Laplacian is the second derivative of the
    image, so it reacts strongly wherever the brightness changes quickly
    (fur, petals, leaves, gravel) and it is almost zero in smooth areas
    (sky, a car door, a plain wall).

    We take the STANDARD DEVIATION of the Laplacian: a smooth image gives a
    small value, a very detailed image gives a large one. (This is the same
    measurement that is normally used to check whether a photo is blurry.)
    """
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    detail = float(laplacian.std())

    return min(detail / LAPLACIAN_STD_REFERENCE, 1.0)


def shape_importance(gray):
    """
    Does this image contain ONE CLEAR, DOMINANT SILHOUETTE?

    That is the question the shape weight must answer. It is NOT the same
    question as "does this image contain a lot of detail" - that is what the
    texture weight measures.

    WHAT WAS WRONG BEFORE
    ---------------------
    The previous version was:

        0.5 * edge_density + 0.5 * main_object_size

    Edge density is the percentage of pixels that Canny marked as an edge.
    On a cat, a dog or a flower, the fur, the petals and the background
    texture produce a huge number of Canny edges. So the very same visual
    detail pushed BOTH the texture importance AND the shape importance up -
    the two weights were measuring the same thing twice.

    Measured on this dataset, the correlation between the old shape
    importance and the texture importance was +0.62, and for edge density
    alone it was +0.70. A shape measure should not follow the texture
    measure that closely.

    THE NEW FORMULA
    ---------------
        shape_importance = 0.50 * area_score      (is there a big object?)
                         + 0.40 * solidity        (is its outline clean?)
                         + 0.10 * edge_score      (small extra contribution)

    a) AREA SCORE - the area of the main contour compared to the whole image.
       This stays, because a silhouette is only meaningful if the object is
       actually big in the picture. We use the SAME contour selection as the
       shape descriptor (choose_main_contour), so the weight describes the
       contour we really measure, not some background blob.

    b) SOLIDITY - this is the new part, and it is what tells a clean
       silhouette apart from a texture blob:

           solidity = contour area / area of its convex hull

       The convex hull is the smallest convex shape that contains the
       contour - imagine stretching a rubber band around the object. A
       compact object (an apple, a car body) fills its rubber band almost
       completely, so solidity is close to 1. A ragged outline full of deep
       notches - exactly what fur, petals and leaves produce - leaves a lot
       of empty space inside the rubber band, so solidity is low.

       Measured average solidity per category on this dataset:
           fruits 0.82,  flowers 0.72,  cars 0.67,  dogs 0.64,  cats 0.53

       Solidity is already a number between 0 and 1, so it needs no
       reference value at all.

    c) EDGE SCORE - edge density is kept, but only with a weight of 0.10
       instead of 0.50, so it can still help a little without dominating.

    The new correlation with the texture importance is +0.53 instead of
    +0.62. It is not zero, because the area score itself grows a little on
    detailed images - but object size is the legitimate core of "is there a
    silhouette here", so we keep it.

    WHY NOT "how dominant is the main contour compared to the others"?
    That sounds like the natural measure, and it was tried first. It does
    not work here: after the Canny edges are dilated almost everything melts
    into ONE single contour (the median number of valid contours per image
    is 1), so there is nothing to compare the main contour against. The
    measure came out as 1.0 for practically every image.
    """
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    total_pixels = float(edges.shape[0] * edges.shape[1])

    # --- c) edge density (only a small contribution now) ---
    edge_density = np.count_nonzero(edges) / total_pixels
    edge_score = min(edge_density / EDGE_DENSITY_REFERENCE, 1.0)

    # --- find the main contour, with the same rules as the shape descriptor ---
    kernel = np.ones((3, 3), dtype=np.uint8)
    closed_edges = cv2.dilate(edges, kernel, iterations=1)
    contours, _ = cv2.findContours(closed_edges, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    main_contour = choose_main_contour(contours, total_pixels)

    # No usable contour -> there is no silhouette, so shape is only worth
    # the small edge contribution.
    if main_contour is None:
        return SHAPE_EDGE_PART * edge_score

    contour_area = cv2.contourArea(main_contour)
    if contour_area <= 0:
        return SHAPE_EDGE_PART * edge_score

    # --- a) how big is the main object ---
    area_ratio = contour_area / total_pixels
    area_score = min(area_ratio / CONTOUR_AREA_REFERENCE, 1.0)

    # --- b) solidity: how much of its convex hull does the contour fill ---
    # cv2.convexHull gives the smallest convex shape around the contour
    # (the "rubber band"). A compact object fills it; a ragged fur or petal
    # outline does not.
    hull = cv2.convexHull(main_contour)
    hull_area = cv2.contourArea(hull)

    if hull_area > 0:
        solidity = contour_area / hull_area
    else:
        solidity = 0.0

    solidity = float(np.clip(solidity, 0.0, 1.0))

    return (SHAPE_AREA_PART * area_score
            + SHAPE_SOLIDITY_PART * solidity
            + SHAPE_EDGE_PART * edge_score)


def local_importance(gray):
    """
    How good are the ORB keypoints of this image?

    Two indicators together:

    a) HOW MANY keypoints ORB found, compared to the maximum it may return.

    b) HOW STRONG those keypoints are on average (the 'response' value of
       ORB). This one matters much more, because ORB almost always returns
       the maximum number of keypoints - on this dataset the median is 285
       out of 300. So the count alone says almost nothing; the response
       separates "285 strong corners" from "285 weak ones".

    THE TRUST FACTOR
    Even a perfect count and a strong response only tell us that the query
    image HAS good keypoints. They cannot tell us whether any database image
    contains the SAME object - and that is the only situation where ORB
    really works. Because the keypoint count is almost always near maximum,
    the old version of this heuristic gave local features 45-50% of the total
    weight for nearly every query, even though ORB is measurably the weakest
    of the three descriptors on this dataset.

    So we multiply the local importance by LOCAL_TRUST (0.4). The local
    weight still changes from query to query, but it can no longer dominate.
    """
    keypoint_count, average_response = orb_keypoint_info(gray)

    count_score = min(keypoint_count / float(ORB_N_FEATURES), 1.0)
    strength_score = min(average_response / ORB_RESPONSE_REFERENCE, 1.0)

    raw_importance = 0.5 * count_score + 0.5 * strength_score

    return raw_importance * LOCAL_TRUST


def compute_weights(gray, return_raw=False):
    """
    Calculate the three importance values for the query image and normalise
    them so that:

        texture_weight + shape_weight + local_weight = 1.0

    Normalising means: divide every value by the sum of all three values.
    After that the three weights can be shown directly as percentages.

    A small MINIMUM_IMPORTANCE is added to each value first, so that no
    descriptor can ever become exactly 0 and disappear completely.

    Returns a dictionary, for example:
        {"texture": 0.41, "shape": 0.34, "local": 0.25}

    With return_raw=True it also returns the three raw importance values,
    which the debug output in database.py prints.
    """
    raw_texture = texture_importance(gray)
    raw_shape = shape_importance(gray)
    raw_local = local_importance(gray)

    texture_value = raw_texture + MINIMUM_IMPORTANCE
    shape_value = raw_shape + MINIMUM_IMPORTANCE
    local_value = raw_local + MINIMUM_IMPORTANCE

    total = texture_value + shape_value + local_value

    # Safety check - 'total' can never be 0 because of MINIMUM_IMPORTANCE,
    # but we check anyway so the program can never divide by zero.
    if total <= 0:
        weights = {"texture": 1.0 / 3.0, "shape": 1.0 / 3.0, "local": 1.0 / 3.0}
    else:
        weights = {
            "texture": texture_value / total,
            "shape": shape_value / total,
            "local": local_value / total,
        }

    if return_raw:
        raw = {"texture": raw_texture, "shape": raw_shape, "local": raw_local}
        return weights, raw

    return weights

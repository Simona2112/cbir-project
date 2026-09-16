"""
distances.py
------------
Here we compare the features of two images.

IMPORTANT: every function in this file returns a DISTANCE, not a similarity.

    distance = 0.0   ->  the two images are identical / very similar
    distance = 1.0   ->  the two images are completely different

So a SMALL distance is GOOD. At the end the results are sorted in ASCENDING
order and the 10 images with the smallest distance are shown.

The three descriptors are compared completely separately - we never glue
them into one big feature vector. Each one has its own distance measure that
fits the kind of data it contains:

    texture (spatial LBP)    ->  Chi-square distance
    shape   (HOG)            ->  Euclidean distance
    local   (ORB or SIFT)    ->  Hamming / L2 distance of the good matches
"""

import cv2
import numpy as np

from config import (ORB_RATIO_TEST, ORB_DESCRIPTOR_BITS,
                    ORB_GOOD_MATCH_REFERENCE, WORST_DISTANCE,
                    SHAPE_REFERENCE_DISTANCE, LBP_GRID, LBP_SCALES,
                    LOCAL_DESCRIPTOR, SIFT_RATIO_TEST,
                    SIFT_GOOD_MATCH_REFERENCE, SIFT_MAX_DISTANCE)

# How many histograms the texture descriptor is made of:
# one per spatial block, per LBP scale (see config.LBP_GRID / LBP_SCALES).
#
#     LBP_GRID = 2, LBP_SCALES = [1]     ->  4 histograms
#     LBP_GRID = 2, LBP_SCALES = [1, 2]  ->  8 histograms
LBP_BLOCK_COUNT = LBP_GRID * LBP_GRID * len(LBP_SCALES)


# =====================================================================
# 1. TEXTURE DISTANCE - Chi-square distance between LBP histograms
# =====================================================================

def texture_distance(hist1, hist2):
    """
    Compare two normalised LBP histograms with the CHI-SQUARE DISTANCE.

        d = 0.5 * sum over all bins of  (h1[i] - h2[i])^2 / (h1[i] + h2[i])

    WHY CHI-SQUARE AND NOT A SIMPLE DIFFERENCE?
    In an LBP histogram a few bins are very large (the codes that appear in
    flat areas) and most bins are small. A plain difference would only look
    at the few big bins and ignore everything else. Chi-square divides every
    bin's difference by how big that bin is, so a difference of 0.01 in a
    small bin counts as much as a difference of 0.01 in a large bin. That is
    exactly what we want, because the small bins describe the interesting
    texture patterns.

    RANGE - AND WHY WE DIVIDE BY THE NUMBER OF BLOCKS
    For ONE histogram that sums to 1.0 this formula is already inside
    [0, 1]: identical histograms give 0.0, and two histograms with no bin in
    common give 0.5 + 0.5 = 1.0.

    But our descriptor is a SPATIAL LBP: it is 4 block histograms glued
    together (see extract_texture_features), and each block was normalised
    on its own, so the whole vector sums to 4, not to 1. Without a
    correction the distance could reach 4.0 and everything would be clipped
    to 1.0, which would destroy the ranking.

    So we divide the sum by the number of blocks. That is exactly the same
    as calculating the chi-square distance of every block separately and
    taking the AVERAGE of the 4 results - which is an easy way to think
    about it - and it puts the distance back inside [0, 1].

    With MULTI-SCALE LBP (LBP_SCALES = [1, 2]) the vector holds 8 blocks:
    4 from the fine scale and 4 from the coarse scale. Dividing by 8 is the
    same as averaging the two scales with equal weight:

        distance = 0.5 * (average of the 4 fine blocks)
                 + 0.5 * (average of the 4 coarse blocks)

    so no separate weights between the scales are needed anywhere.

    (With LBP_GRID = 1 there is a single block, the division is by 1, and
    the formula is the classic global version again.)

    The small 1e-10 in the denominator prevents a division by zero when both
    histograms are 0 in the same bin.
    """
    if hist1 is None or hist2 is None:
        return WORST_DISTANCE

    numerator = (hist1 - hist2) ** 2
    denominator = hist1 + hist2 + 1e-10

    distance = 0.5 * float(np.sum(numerator / denominator))

    # average over all blocks of all scales (see the explanation above)
    distance = distance / LBP_BLOCK_COUNT

    return float(np.clip(distance, 0.0, 1.0))


# =====================================================================
# 2. SHAPE DISTANCE - Euclidean distance between HOG vectors
# =====================================================================

def shape_distance(shape1, shape2):
    """
       Compare two HOG shape descriptors using Euclidean distance.

       HOG describes the distribution of edge orientations in local
       regions of the image.

       Both HOG vectors are L2-normalized, so Euclidean distance is used:

       d = sqrt(sum((a[i] - b[i])^2))

       A smaller distance means that the two images have more similar
       edge and shape structure.
    """
    if shape1 is None or shape2 is None:
        return WORST_DISTANCE

    difference = shape1.astype(np.float64) - shape2.astype(np.float64)
    raw_distance = float(np.sqrt(np.sum(difference * difference)))

    return float(np.clip(raw_distance / SHAPE_REFERENCE_DISTANCE, 0.0, 1.0))


# =====================================================================
# 3. LOCAL FEATURE DISTANCE - ORB/SIFT descriptor matching
# =====================================================================

# The matchers are created once and reused.
#
# NORM_HAMMING is the correct distance for the BINARY ORB descriptors:
# it counts how many of the 256 bits are different.
#
# NORM_L2 is the correct distance for the FLOATING POINT SIFT descriptors:
# it is the ordinary straight-line distance between two 128-number vectors.
brute_force_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
sift_matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)


def local_distance(descriptors1, descriptors2):
    """
    Compare the ORB descriptors of two images and return a DISTANCE.

    Steps:
        1. knnMatch(k=2) finds, for every descriptor of image 1, its two
           closest descriptors in image 2 (closest in Hamming distance).

        2. LOWE'S RATIO TEST keeps a match only if the best candidate is
           clearly better than the second best:

               best.distance < 0.75 * second_best.distance

           If the two candidates are almost equally good, the keypoint looks
           like many other points and the match cannot be trusted, so we
           throw it away. This removes most of the wrong matches.

        3. QUALITY: the mean Hamming distance of the good matches, divided
           by 256 (an ORB descriptor has 256 bits). 0.0 means the matched
           descriptors were bit-for-bit identical.

        4. CONFIDENCE: the mean is only meaningful if it was computed from
           enough matches. With one lucky match the mean can be very small
           by pure chance, and that image would jump to the first place.
           So we measure how much we trust the mean:

               confidence = min(good_matches / 10, 1.0)

           and blend the measured quality with the worst possible distance:

               local_distance = confidence * quality + (1 - confidence) * 1.0

           With 10 or more good matches we fully trust the measurement.
           With 2 good matches we only trust it 20%, so the distance stays
           close to 1.0 ("we did not really find this object again").

        5. If there are no descriptors at all, or no good matches, we return
           the worst distance 1.0 instead of crashing.

    Why this is better than the old version: the old code returned
    "good_matches / number_of_descriptors", which was about 0.005 for almost
    every pair of images, so the local score was practically always 0 and
    carried no information.
    """
    # An image can have no descriptors at all (for example a plain sky).
    if descriptors1 is None or descriptors2 is None:
        return WORST_DISTANCE
    if len(descriptors1) < 2 or len(descriptors2) < 2:
        return WORST_DISTANCE

    # Pick the matcher, the ratio and the scale that belong to the descriptor
    # we are using. Everything below is then exactly the same for both.
    if LOCAL_DESCRIPTOR == "sift":
        matcher = sift_matcher                 # L2, for float descriptors
        ratio_test = SIFT_RATIO_TEST
        largest_possible = SIFT_MAX_DISTANCE   # 512 * sqrt(2) = 724.1
        match_reference = SIFT_GOOD_MATCH_REFERENCE
    else:
        matcher = brute_force_matcher          # Hamming, for binary descriptors
        ratio_test = ORB_RATIO_TEST
        largest_possible = ORB_DESCRIPTOR_BITS  # 256 bits
        match_reference = ORB_GOOD_MATCH_REFERENCE

    try:
        matches = matcher.knnMatch(descriptors1, descriptors2, k=2)
    except cv2.error:
        return WORST_DISTANCE

    # --- Lowe's ratio test: collect the distances of the good matches
    good_distances = []
    for match_pair in matches:
        # knnMatch can return fewer than 2 neighbours, then we skip the pair
        if len(match_pair) < 2:
            continue
        best, second_best = match_pair
        if best.distance < ratio_test * second_best.distance:
            good_distances.append(best.distance)

    # no good match at all -> the images have nothing in common
    if len(good_distances) == 0:
        return WORST_DISTANCE

    # --- how good are the matches (0.0 = identical descriptors)
    # Dividing by the LARGEST POSSIBLE distance of that descriptor puts ORB
    # and SIFT on exactly the same 0..1 scale, so switching the descriptor
    # cannot make the local distance suddenly bigger or smaller than the
    # other two distances.
    quality = float(np.mean(good_distances)) / largest_possible

    # --- how much do we trust that number
    confidence = min(len(good_distances) / float(match_reference), 1.0)

    distance = confidence * quality + (1.0 - confidence) * WORST_DISTANCE

    return float(np.clip(distance, 0.0, 1.0))


# =====================================================================
# COMBINING THE THREE DISTANCES
# =====================================================================

def normalise_by_mean(distance_list):
    """
    Put one list of distances on a comparable scale.

    WHY THIS IS NEEDED
    The three distances are mathematically all inside [0, 1], but in practice
    they use very different parts of that range. Measured on this dataset:

        texture (chi-square) : typical value about 0.05
        shape (Euclidean)    : typical value about 0.53
        local (ORB)          : typical value about 0.84

    If we put those numbers straight into a weighted average, shape and local
    would decide the ranking no matter what the weights say, simply because
    their numbers are about 16 times bigger. The texture weight could be 90%
    and it would still hardly change the result.

    THE FIX
    For one query we divide every distance by the AVERAGE distance of that
    same descriptor to all database images:

        normalised[i] = distance[i] / mean(all distances)

    After this, 1.0 means "an average image", below 1.0 means "closer than
    average" and above 1.0 means "further than average". All three descriptors
    now speak the same language, so the weights really control the balance.

    This is done per query, which is fine because the ranking only compares
    images inside one query.
    """
    values = np.asarray(distance_list, dtype=np.float64)

    mean_value = values.mean()
    if mean_value <= 1e-12:
        # all distances are 0 (for example a database with one image that is
        # identical to the query) - nothing to normalise
        return values

    return values / mean_value


def combine_distances(texture_values, shape_values, local_values, weights):
    """
    Combine the three normalised distances into ONE final distance, using the
    weights that were calculated automatically from the query image.

        final_distance = texture_weight * texture_distance
                       + shape_weight   * shape_distance
                       + local_weight   * local_distance

    The three weights sum to 1.0, so the final distance stays on the same
    scale as the three normalised distances (around 1.0 for an average image).

    LOWER final_distance = MORE SIMILAR.

    The input lists are the distances of the query to EVERY database image,
    so the normalisation above can use the average over the whole database.
    """
    texture_normalised = normalise_by_mean(texture_values)
    shape_normalised = normalise_by_mean(shape_values)
    local_normalised = normalise_by_mean(local_values)

    final = (weights["texture"] * texture_normalised
             + weights["shape"] * shape_normalised
             + weights["local"] * local_normalised)

    return final, texture_normalised, shape_normalised, local_normalised

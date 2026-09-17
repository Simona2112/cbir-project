"""
config.py

Main project settings for the CBIR application.

The project works with distances:
smaller distance = more similar images.
"""

import os
import math


# ============================================================
# PROJECT PATHS
# ============================================================

# Folder where this file is located
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

# Dataset folder
DATASET_DIR = os.path.join(PROJECT_DIR, "dataset")

# Folder and file where extracted features are stored
FEATURES_DIR = os.path.join(PROJECT_DIR, "features")
FEATURES_FILE = os.path.join(FEATURES_DIR, "features.pkl")

# Change this when the feature format changes,
# so the feature index can be rebuilt automatically.
FEATURES_FORMAT_VERSION = 7


# ============================================================
# GENERAL IMAGE SETTINGS
# ============================================================

# All images are preprocessed to the same size
IMAGE_SIZE = 256

# Supported image formats
VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")

# Use all images from every dataset category
MAX_IMAGES_PER_CLASS = None

# Images with very low contrast are ignored
MIN_CONTRAST_STD = 12.0

# Number of similar images returned by the search
TOP_K = 10


# ============================================================
# TEXTURE DESCRIPTOR - SPATIAL LBP
# ============================================================

# Divide the image into a 2x2 grid.
# Every block has its own 256-bin LBP histogram.
LBP_GRID = 2

# 8-neighbour LBP produces values from 0 to 255
LBP_BINS = 256

# Radius used for LBP.
# [1] means the standard single-scale LBP version.
LBP_SCALES = [1]


# ============================================================
# SHAPE DESCRIPTOR - HOG
# ============================================================

# Final version uses HOG.
# "hu" can still be used for comparison if needed.
SHAPE_DESCRIPTOR = "hog"

# HOG settings
# 256 / 64 = 4, so the image is divided into a 4x4 cell grid.
HOG_CELL_SIZE = 64

# Number of orientation bins in every HOG cell
HOG_BINS = 9

# Parameters used only by the old Hu-moments implementation
SHAPE_MIN_CONTOUR_RATIO = 0.01
SHAPE_MAX_CONTOUR_RATIO = 0.90

# Reference value used to normalize the shape distance
if SHAPE_DESCRIPTOR == "hog":
    SHAPE_REFERENCE_DISTANCE = math.sqrt(2.0)
else:
    SHAPE_REFERENCE_DISTANCE = 1.5


# ============================================================
# REGION OF INTEREST
# ============================================================

# ROI was tested but is disabled in the final version
USE_ROI = False

# ROI settings kept in case the experiment is repeated
ROI_MARGIN = 0.10
ROI_MIN_FRACTION = 0.05
ROI_MAX_FRACTION = 0.90
ROI_MIN_SIDE = 16


# ============================================================
# LOCAL FEATURES - ORB
# ============================================================

# Final version uses ORB.
# "sift" is still supported for comparison.
LOCAL_DESCRIPTOR = "orb"

# Maximum number of ORB keypoints per image
ORB_N_FEATURES = 300

# Lowe's ratio-test threshold
ORB_RATIO_TEST = 0.75

# ORB descriptor contains 256 bits
ORB_DESCRIPTOR_BITS = 256.0

# Number of good matches needed for full confidence
ORB_GOOD_MATCH_REFERENCE = 10

# Returned when no valid comparison can be made
WORST_DISTANCE = 1.0


# ============================================================
# SIFT SETTINGS
# ============================================================

# SIFT is not used in the final version,
# but the settings are kept for comparison experiments.
SIFT_N_FEATURES = 300
SIFT_RATIO_TEST = 0.75
SIFT_GOOD_MATCH_REFERENCE = 10

# Maximum L2 distance used for normalization
SIFT_MAX_DISTANCE = 512.0 * math.sqrt(2.0)


# ============================================================
# AUTOMATIC DESCRIPTOR WEIGHTS
# ============================================================

# Texture importance reference
LAPLACIAN_STD_REFERENCE = 62.0

# Shape importance references
EDGE_DENSITY_REFERENCE = 0.15
CONTOUR_AREA_REFERENCE = 0.65

# Parts used when calculating shape importance
SHAPE_AREA_PART = 0.50
SHAPE_SOLIDITY_PART = 0.40
SHAPE_EDGE_PART = 0.10

# Local-feature importance reference
ORB_RESPONSE_REFERENCE = 0.0026

# ORB is useful, but it should not dominate the final score
LOCAL_TRUST = 0.4

# Every descriptor keeps at least a small influence
MINIMUM_IMPORTANCE = 0.05
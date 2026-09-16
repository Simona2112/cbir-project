"""
config.py
---------
All the settings of the project in one place, so they are easy to find
and easy to change before a demo.

NOTE: this project works with DISTANCES, not with similarities.
      A SMALL distance means the two images are SIMILAR.
"""

import os
import math

# Folder of this file (so the program works no matter where it is started from)
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

# Where the images are
DATASET_DIR = os.path.join(PROJECT_DIR, "dataset")

# Where the calculated features are saved
FEATURES_DIR = os.path.join(PROJECT_DIR, "features")
FEATURES_FILE = os.path.join(FEATURES_DIR, "features.pkl")

# Version number of the saved feature file.
# If we change how the descriptors are calculated, an old features.pkl would
# contain features that do not match the new code. We store this number inside
# the file; when it does not match, the index is rebuilt automatically.
FEATURES_FORMAT_VERSION = 7

# Every image is resized to this size before we calculate features.
# Using the same size for all images makes the features comparable.
IMAGE_SIZE = 256

# File types we accept
VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")


# =====================================================================
# TEXTURE DESCRIPTOR - SPATIAL LBP
# =====================================================================

# The image is divided into a LBP_GRID x LBP_GRID grid of blocks, and every
# block gets its own 256-bin LBP histogram. The histograms are then joined
# into one long feature vector:
#
#     LBP_GRID = 1  ->  1 block   ->  1 * 256 =  256 numbers  (one global histogram)
#     LBP_GRID = 2  ->  4 blocks  ->  4 * 256 = 1024 numbers
#     LBP_GRID = 4  -> 16 blocks  -> 16 * 256 = 4096 numbers
#
# WHY A GRID AT ALL?
# One global histogram only says WHICH textures are in the image, never
# WHERE. Splitting the image into blocks adds coarse position information.
#
# WHICH VALUE IS BEST?
# We measured all of them on this dataset (texture-only Precision@10):
#
#     1x1 (global) : 0.436
#     2x2          : 0.439   <- best
#     4x4          : 0.414
#     8x8          : 0.413
#
# 2x2 is used because it is the only setting that beats the global
# histogram. A finer grid makes things WORSE here, because spatial LBP
# assumes the object is in the same place in every photo (that is why it
# works so well for faces, where the eyes and mouth are always aligned).
# In this dataset a cat can be anywhere in the frame, so a fine grid mostly
# adds position noise. A fine grid also leaves few pixels per histogram
# bin, which makes the chi-square distance noisy.
#
# Change this single number to try another grid - nothing else has to be
# touched, and the feature file is rebuilt automatically.
LBP_GRID = 2

# Number of bins of one LBP histogram (an 8-neighbour LBP code is 0..255)
LBP_BINS = 256

# MULTI-SCALE LBP
# ---------------
# Basic LBP only looks at the 8 pixels touching the centre, so it describes
# texture at ONE very small scale. Two different materials can look similar
# that close up. By sampling the 8 neighbours further away we describe a
# COARSER texture structure:
#
#     radius = 1  ->  fine texture   (the pixels touching the centre)
#     radius = 2  ->  coarser texture (the pixels 2 steps away)
#
# Every radius in this list produces its own complete spatial descriptor,
# and they are concatenated:
#
#     LBP_SCALES = [1]      ->  4 blocks * 256          = 1024 numbers
#     LBP_SCALES = [1, 2]   ->  2 * 4 blocks * 256      = 2048 numbers
#
# WAS IT WORTH IT? (measured on this dataset - texture-only / combined P@10)
#
#     [1]        (fine only)    0.439 / 0.388     <- used
#     [2]        (coarse only)  0.419 / 0.368
#     [1, 2]     (multi-scale)  0.431 / 0.375
#     [1, 2, 3]                 0.433 / 0.372
#
# Multi-scale was tested and NOT kept: the coarse scale is weaker on its own,
# so averaging it in pulls the fine scale down instead of adding information.
# The code supports it, so the experiment can be repeated by changing this
# one line - but [1] is what actually performs best here.
#
# Set it back to [1] to get exactly the single-scale version again.
LBP_SCALES = [1]

# How many images we take from each folder.
# The current dataset has 40 images in each of the 5 folders (200 in total),
# which is small and fast, so we simply use all of them.
# Set this to a number (for example 20) if you want a smaller demo index.
MAX_IMAGES_PER_CLASS = None

# Minimum contrast an image must have to be useful.
# It is the standard deviation of the grayscale values BEFORE CLAHE.
# A picture where every pixel has almost the same value carries no
# information; after contrast stretching it becomes pure noise that matches
# everything. See preprocessing.has_enough_contrast().
MIN_CONTRAST_STD = 12.0

# How many results we show
TOP_K = 10


# =====================================================================
# SHAPE DESCRIPTOR
# =====================================================================

# Which contours may be chosen as the "main object".
# A contour smaller than 1% of the image is noise (a leaf, a stone).
# A contour bigger than 90% of the image is usually the background or the
# border of the picture itself, not the object we are looking for.
SHAPE_MIN_CONTOUR_RATIO = 0.01
SHAPE_MAX_CONTOUR_RATIO = 0.90

# WHICH SHAPE DESCRIPTOR IS USED
# ------------------------------
#   "hog" - Histogram of Oriented Gradients (the default)
#   "hu"  - the older Canny + contour + Hu moments descriptor
#
# Both are implemented; this switch decides which one the program uses.
# Measured on this dataset (shape-only / combined Precision@10):
#
#     "hu"   0.268 / 0.387
#     "hog"  0.449 / 0.483     <- clearly better
#
# The difference is far bigger than the measurement noise (paired over the
# 200 queries: shape-only +0.153 +/- 0.023, combined +0.079 +/- 0.015), so
# HOG is the one we use. See README.md.
SHAPE_DESCRIPTOR = "hog"

# --- HOG settings ---
# The image is cut into cells of HOG_CELL_SIZE x HOG_CELL_SIZE pixels, and
# every cell gets a histogram of edge directions with HOG_BINS bins.
#
#     256 / 64 = 4   ->  a 4 x 4 grid of cells
#     4 * 4 * 9 = 144 numbers
#
# We measured the cell size (shape-only Precision@10):
#
#     16 px (2304 numbers) : 0.390
#     32 px ( 576 numbers) : 0.420
#     64 px ( 144 numbers) : 0.449   <- used
#    128 px (  36 numbers) : 0.386
#
# It gets better up to 64 and worse again at 128, so 64 is a real optimum
# and not simply "coarser is always better".
HOG_CELL_SIZE = 64
HOG_BINS = 9

# How the shape distance is brought into the range 0..1.
#
# For HOG we L2-normalise the descriptor, so it is a unit vector. All HOG
# values are >= 0, so two such vectors can at worst be at right angles to
# each other, and then their Euclidean distance is exactly sqrt(2). That is
# the largest distance that can ever occur, so dividing by sqrt(2) is not a
# tuned number - it follows from the mathematics.
#
# For the Hu descriptor the 12 numbers lie in [-1, 1] and [0, 1], where the
# theoretical maximum is 5.74 but real photographs almost never go above
# 1.5, so 1.5 is used there.
if SHAPE_DESCRIPTOR == "hog":
    SHAPE_REFERENCE_DISTANCE = math.sqrt(2.0)
else:
    SHAPE_REFERENCE_DISTANCE = 1.5


# =====================================================================
# REGION OF INTEREST (ROI)
# =====================================================================
# When this is True, the texture (LBP) and the local (ORB) features are not
# taken from the whole picture, but only from the area around the main
# object: we find the main contour, take its bounding rectangle, make it a
# little bigger, and cut that part out. The idea is that a dog standing on
# grass should be described by the dog, not by the grass.
#
# The shape descriptor is NOT affected - it already works on the main
# contour itself.
#
# If no reliable region is found, the full image is used (see
# feature_extractor.extract_main_roi).
#
# WAS IT WORTH IT? Measured on this dataset (200 queries):
#
#                      texture  ORB    combined  P@1
#     USE_ROI = False   0.439   0.264   0.387    0.500   <- used
#     USE_ROI = True    0.444   0.249   0.394    0.495
#
# The combined difference (+0.007) is smaller than the measurement noise
# (standard error +/- 0.007, t = 1.1), so it is NOT a real improvement, and
# Precision@1 even went down slightly. The main reason is that on these
# photos the main contour is usually already almost the whole picture: the
# region covered 82% of the image on average, so very little background was
# actually removed. ORB clearly got worse, because a cropped and re-enlarged
# image has fewer clean corners.
#
# The experiment is kept in the code so it can be repeated - set this to
# True and the feature index rebuilds itself - but False is what is used.
USE_ROI = False

# How much bigger than the bounding rectangle the region is made,
# on each side (0.10 = 10%), so the object is not cut off exactly at its edge.
ROI_MARGIN = 0.10

# A region is only accepted if it covers between 5% and 90% of the image.
# Smaller means the contour found some little detail, not the object;
# larger means the region is basically the whole picture, so cropping it
# would not remove any background anyway.
ROI_MIN_FRACTION = 0.05
ROI_MAX_FRACTION = 0.90

# Smallest side (in pixels) a region must have to be usable at all.
ROI_MIN_SIDE = 16


# =====================================================================
# LOCAL FEATURES (ORB)
# =====================================================================

# WHICH LOCAL DESCRIPTOR IS USED
# ------------------------------
#   "orb"  - ORB: binary descriptors, compared with the Hamming distance
#   "sift" - SIFT: floating point descriptors, compared with the L2 distance
#
# Both are implemented; this switch decides which one the program uses.
# Changing it rebuilds the feature index automatically.
#
# WAS SIFT BETTER? Measured on this dataset (200 queries):
#
#                local-only  combined  P@1
#     "orb"        0.264      0.482    0.530   <- used
#     "sift"       0.251      0.483    0.540
#
# SIFT was tested and NOT kept. It did not improve the local descriptor:
# local-only retrieval actually got slightly WORSE, and the combined result
# moved by +0.001, which is far inside the measurement noise (paired over
# the 200 queries: +0.001 +/- 0.009, t = 0.1).
#
# The reason is that the problem is not the descriptor, it is the task. Both
# ORB and SIFT only work well when the SAME object appears in two photos.
# On this dataset the median pair has just 2 good matches after Lowe's ratio
# test with either descriptor - two different cats are simply two different
# objects. SIFT is also about 4x slower to compute and match.
LOCAL_DESCRIPTOR = "orb"

ORB_N_FEATURES = 300      # maximum number of keypoints per image
ORB_RATIO_TEST = 0.75     # Lowe's ratio test threshold for "good" matches

# An ORB descriptor is 32 bytes = 256 bits, so the Hamming distance between
# two descriptors is a number between 0 and 256. We divide by 256 to bring
# the local distance into the range 0..1.
ORB_DESCRIPTOR_BITS = 256.0

# How many good matches we need before we trust the measured match quality.
# With only 1 or 2 good matches the mean Hamming distance is pure luck, so
# we push the distance towards the worst value instead (see distances.py).
ORB_GOOD_MATCH_REFERENCE = 10

# The value we return when two images cannot be compared at all
# (no descriptors, or no good matches). 1.0 = "as different as possible".
WORST_DISTANCE = 1.0


# --- SIFT settings (used when LOCAL_DESCRIPTOR = "sift") ---

SIFT_N_FEATURES = 300      # same budget as ORB, so the comparison is fair
SIFT_RATIO_TEST = 0.75     # Lowe's ratio test, same value as for ORB
SIFT_GOOD_MATCH_REFERENCE = 10   # same "confidence" rule as ORB

# The largest L2 distance that can occur between two SIFT descriptors.
#
# OpenCV returns SIFT descriptors as unit vectors multiplied by 512, so every
# descriptor has a length of about 512 (measured on this dataset: median
# 512.0, 5th percentile 511.1, 95th percentile 512.9). All values are >= 0,
# so two descriptors can at worst stand at right angles to each other, and
# then their distance is 512 * sqrt(2).
#
# This is the exact counterpart of ORB_DESCRIPTOR_BITS = 256 (the largest
# Hamming distance between two 256-bit descriptors). Dividing by it puts the
# SIFT distance on the SAME 0..1 scale as the ORB one, so switching the
# descriptor cannot make the local distance suddenly dominate the final
# score just because raw L2 numbers are bigger than Hamming numbers.
SIFT_MAX_DISTANCE = 512.0 * math.sqrt(2.0)


# =====================================================================
# AUTOMATIC WEIGHTS (see weighting.py)
# =====================================================================
# These reference values are roughly the 90th percentile of what we measured
# on this dataset. A typical image therefore lands in the middle of the range
# and only an extreme image reaches 1.0. If they were too small, every image
# would reach 1.0, all three weights would always be equal, and the dynamic
# weighting would do nothing.

LAPLACIAN_STD_REFERENCE = 62.0     # texture: how much fine detail
EDGE_DENSITY_REFERENCE = 0.15      # shape: percentage of Canny edge pixels
CONTOUR_AREA_REFERENCE = 0.65      # shape: size of the main object

# How the shape importance is put together (the three parts add up to 1.0).
#
# Edge density used to be half of the shape importance, but it is really a
# measure of general detail: fur, petals and background texture produce a lot
# of Canny edges, so the same detail was raising the texture weight AND the
# shape weight at the same time. It is kept, but only with a small share.
#
# Solidity (how much of its convex hull the main contour fills) is what
# actually separates a clean, compact silhouette from a ragged texture blob.
# See the long explanation in weighting.shape_importance().
SHAPE_AREA_PART = 0.50       # is there a big main object at all?
SHAPE_SOLIDITY_PART = 0.40   # is its outline clean rather than ragged?
SHAPE_EDGE_PART = 0.10       # small contribution from edge density
ORB_RESPONSE_REFERENCE = 0.0026    # local: average keypoint strength

# How much we are willing to trust local features.
#
# ORB almost always returns the maximum number of keypoints (on this dataset
# the median is 285 out of 300), so a weight based on the keypoint count alone
# gave local features 45-50% of the total weight. But ORB only works when the
# SAME object appears in both photographs, which is rare in a general image
# dataset - measured on its own it is the weakest of the three descriptors
# (see the results table in README.md).
#
# This factor multiplies the local importance before normalising, so local
# features can still vary from query to query but can never dominate.
LOCAL_TRUST = 0.4

# Every descriptor gets this minimum importance, so no descriptor can ever
# drop to exactly 0 and disappear completely.
MINIMUM_IMPORTANCE = 0.05

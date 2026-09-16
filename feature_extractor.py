"""
feature_extractor.py
--------------------
Here we calculate the three descriptors of the project:

    1. TEXTURE  -> Local Binary Pattern (LBP) histogram
    2. SHAPE    -> HOG (Histogram of Oriented Gradients)
                   (the older Canny + contour + Hu moments descriptor is
                    still here, selected with SHAPE_DESCRIPTOR in config.py)
    3. LOCAL    -> ORB keypoints and descriptors
                   (SIFT is also implemented, selected with LOCAL_DESCRIPTOR
                    in config.py)

All of them work on the GRAYSCALE image, so colour never influences
the retrieval (this is a requirement of the project).
"""

import cv2
import numpy as np

from config import (ORB_N_FEATURES, SHAPE_MIN_CONTOUR_RATIO,
                    SHAPE_MAX_CONTOUR_RATIO, LBP_GRID, LBP_BINS,
                    LBP_SCALES, USE_ROI, ROI_MARGIN, ROI_MIN_FRACTION,
                    ROI_MAX_FRACTION, ROI_MIN_SIDE, IMAGE_SIZE,
                    SHAPE_DESCRIPTOR, HOG_CELL_SIZE, HOG_BINS,
                    LOCAL_DESCRIPTOR, SIFT_N_FEATURES)

# We create the ORB detector only once and reuse it for every image.
# Creating it for every image would be slow.
orb_detector = cv2.ORB_create(nfeatures=ORB_N_FEATURES)

# The SIFT detector, also created only once (used when LOCAL_DESCRIPTOR="sift")
sift_detector = cv2.SIFT_create(nfeatures=SIFT_N_FEATURES)


# =====================================================================
# 1. TEXTURE DESCRIPTOR - Local Binary Pattern (LBP)
# =====================================================================

def compute_lbp_image(gray, radius=1):
    """
    Calculate the LBP image "by hand" with NumPy.

    IDEA OF LBP
    -----------
    For every pixel we look at its 8 neighbours. For each neighbour we ask
    one simple question: "is the neighbour brighter or equal to the centre?"

        yes -> write 1
        no  -> write 0

    Reading those 8 bits in a fixed order gives a binary number between
    00000000 and 11111111, so a value between 0 and 255. That number is
    the LBP code of the pixel.

        neighbour layout          bit weights
        -----------------         ------------
        p0  p1  p2                  1    2    4
        p7   C  p3                128    C    8
        p6  p5  p4                 64   32   16

    A flat area gives a code with few changes, an edge or a rough texture
    gives a very different code. So the LBP code describes the LOCAL TEXTURE
    around the pixel, and it does not care about the overall brightness of
    the image (only about differences between neighbours).

    THE RADIUS (multi-scale LBP)
    ----------------------------
    'radius' says how FAR AWAY the 8 neighbours are taken from. With
    radius = 1 we use the 8 pixels that touch the centre, which is the
    classic LBP. With radius = 2 we use the 8 pixels on a square ring two
    steps away - the same 8 directions, only further out:

        radius = 1                    radius = 2

                                 p0 .  p1 .  p2
        p0 p1 p2                 .  .  .  .  .
        p7  C p3                 p7 .  C  .  p3
        p6 p5 p4                 .  .  .  .  .
                                 p6 .  p5 .  p4

    A bigger radius describes a COARSER texture. All 8 sample points still
    land exactly on real pixels, so we never have to interpolate between
    pixels - that is what keeps this implementation simple. (The textbook
    "circular LBP" puts the points on a real circle and has to interpolate;
    we do not need that here.)

    HOW IT IS DONE WITH NUMPY
    -------------------------
    Instead of two slow Python 'for' loops over every pixel, we take the
    whole neighbour image at once by SHIFTING the image. 'padded[0:-2, 0:-2]'
    is the image of all top-left neighbours, and so on. Then one single
    comparison handles all pixels at the same time. This is much faster and
    it is the normal way of working with NumPy.
    """
    height, width = gray.shape

    # Add a border of 'radius' pixels so that the pixels at the edge of the
    # image also have 8 neighbours.
    # BORDER_REPLICATE copies the closest pixel value into the border.
    padded = cv2.copyMakeBorder(gray, radius, radius, radius, radius,
                                cv2.BORDER_REPLICATE)

    # Use int16 so the comparison is done on real numbers, not on uint8
    padded = padded.astype(np.int16)

    # The centre pixels = the original image
    centre = padded[radius:radius + height, radius:radius + width]

    # Where the 8 neighbours are, as (row offset, column offset).
    # The order is the same as the bit weights 1, 2, 4, ... 128.
    offsets = [
        (-radius, -radius),   # p0 top-left      weight 1
        (-radius,       0),   # p1 top           weight 2
        (-radius,  radius),   # p2 top-right     weight 4
        (      0,  radius),   # p3 right         weight 8
        ( radius,  radius),   # p4 bottom-right  weight 16
        ( radius,       0),   # p5 bottom        weight 32
        ( radius, -radius),   # p6 bottom-left   weight 64
        (      0, -radius),   # p7 left          weight 128
    ]

    lbp = np.zeros(centre.shape, dtype=np.uint8)

    # Build the binary number bit by bit.
    # Instead of two slow Python loops over every pixel, we take the whole
    # neighbour image at once by SHIFTING the padded image.
    for bit_index, (row_offset, column_offset) in enumerate(offsets):
        neighbour = padded[radius + row_offset:radius + row_offset + height,
                           radius + column_offset:radius + column_offset + width]

        # 1 where the neighbour is >= the centre, 0 otherwise
        bit = (neighbour >= centre).astype(np.uint8)
        # Put that bit on its position (1, 2, 4, 8, ... 128) and add it
        lbp = lbp + bit * (1 << bit_index)

    return lbp


def extract_texture_features(gray):
    """
    Turn the LBP image into a feature vector - SPATIAL LBP.

    THE PROBLEM WITH ONE GLOBAL HISTOGRAM
    -------------------------------------
    The simplest version counts every LBP code of the whole image into ONE
    histogram. That tells us WHICH textures the image contains and in what
    proportion, but nothing about WHERE they are. A stripe in the top-left
    corner and the same stripe in the bottom-right corner land in exactly
    the same bin.

    SPATIAL LBP
    -----------
    So we cut the image into a grid of LBP_GRID x LBP_GRID blocks and give
    every block its OWN histogram. Then we put all the histograms one after
    another into a single long vector:

        LBP_GRID = 2  ->  4 blocks of 128x128  ->  4 * 256 = 1024 numbers

        +---------+---------+
        |  hist 0 |  hist 1 |      the final vector is
        +---------+---------+      [ hist0 | hist1 | hist2 | hist3 ]
        |  hist 2 |  hist 3 |
        +---------+---------+

    Now the descriptor knows that "this texture was in the upper half", so
    two images only match well if they have similar textures in similar
    places.

    TWO IMPORTANT DETAILS
    ---------------------
    1. The LBP is calculated ONCE on the complete image and only then cut
       into blocks. If we ran the LBP separately on each block, the pixels
       on the block borders would have no real neighbours and OpenCV would
       invent them by copying - that would put fake texture into every
       block edge.

    2. Every block histogram is divided by the number of pixels of THAT
       block, so each one sums to 1.0 on its own. This means the whole
       vector sums to (number of blocks), not to 1.0 - the chi-square
       distance in distances.py takes care of that by dividing by the
       number of blocks.

    MULTI-SCALE
    -----------
    The whole thing is repeated for every radius in LBP_SCALES, and the
    results are put one after another:

        LBP_SCALES = [1]     ->  4 blocks * 256      = 1024 numbers
        LBP_SCALES = [1, 2]  ->  2 * 4 * 256         = 2048 numbers

    Radius 1 describes fine texture, radius 2 a coarser structure.

    Returns len(LBP_SCALES) * LBP_GRID * LBP_GRID * 256 numbers.
    """
    histograms = []

    # one complete spatial descriptor per scale
    for radius in LBP_SCALES:

        # --- calculate the LBP image once, for the whole picture
        lbp = compute_lbp_image(gray, radius=radius)

        height, width = lbp.shape
        block_height = height // LBP_GRID
        block_width = width // LBP_GRID

        # go through the blocks row by row
        for row in range(LBP_GRID):
            for column in range(LBP_GRID):
                # cut out one block of the LBP image
                y_start = row * block_height
                x_start = column * block_width
                block = lbp[y_start:y_start + block_height,
                            x_start:x_start + block_width]

                # np.bincount counts the occurrences of every code 0..255
                histogram = np.bincount(block.ravel(),
                                        minlength=LBP_BINS).astype(np.float32)

                # normalise THIS block, so it sums to 1.0
                total = histogram.sum()
                if total > 0:
                    histogram = histogram / total

                histograms.append(histogram)

    # glue all block histograms of all scales into one long feature vector
    return np.concatenate(histograms)


# =====================================================================
# 2. SHAPE DESCRIPTOR - Canny + contours + Hu moments
# =====================================================================

def choose_main_contour(contours, image_area):
    """
    Pick the contour that most probably belongs to the main object.

    THE PROBLEM WITH "JUST TAKE THE BIGGEST ONE"
    On a natural photograph Canny does not only find the outline of the cat -
    it also finds grass, leaves, clouds and the edge of the picture itself.
    Very often the biggest contour is therefore the BACKGROUND, or even a
    rectangle around the whole image, and then the Hu moments describe a
    rectangle instead of the object.

    THREE SIMPLE RULES
    We throw a contour away if:

        1. it is smaller than 1% of the image  -> it is noise (a stone, a leaf)
        2. it is bigger than 90% of the image  -> it is the background, not an object
        3. its bounding box covers the whole image -> it is the picture border

    From the contours that survive we take the biggest one, because among the
    real candidates the biggest is the most important object.

    If no contour survives (for example a very busy photo where everything is
    connected), we fall back to the biggest contour of all, so the program
    always returns something instead of failing.
    """
    if len(contours) == 0:
        return None

    # the side length of the (square) processed image
    image_side = int(round(np.sqrt(image_area)))

    good_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)

        # rule 1 and 2: too small or too big
        if area < SHAPE_MIN_CONTOUR_RATIO * image_area:
            continue
        if area > SHAPE_MAX_CONTOUR_RATIO * image_area:
            continue

        # rule 3: the bounding box is (almost) the whole image -> it is the border
        x, y, width, height = cv2.boundingRect(contour)
        if x <= 1 and y <= 1 and width >= image_side - 2 and height >= image_side - 2:
            continue

        good_contours.append(contour)

    if len(good_contours) == 0:
        # nothing survived -> fall back to the biggest contour
        return max(contours, key=cv2.contourArea)

    return max(good_contours, key=cv2.contourArea)


def extract_shape_features(gray):
    """
    Describe the SHAPE of the main object in the image.

    Steps:
        1. Gaussian blur  -> removes noise, otherwise Canny finds fake edges
        2. Canny          -> finds the edges (object boundaries)
        3. dilate         -> closes small gaps so the contour is not broken
        4. findContours   -> groups the edge pixels into closed curves
        5. choose_main_contour -> pick the contour that really is the object
        6. Hu moments     -> 7 numbers that describe the shape of that contour

    HU MOMENTS
    ----------
    Hu moments are 7 numbers calculated from the moments of the shape.
    Their special property is that they do not change when the shape is
    moved, rotated or scaled. That is exactly what we want: a tiger seen
    from the left and the same tiger seen bigger should give similar values.

    The raw Hu values are extremely small (like 1e-07), so we compress them
    with a logarithm - this is the standard way of using Hu moments.

    We also add 5 simple geometric features (area, perimeter, circularity,
    aspect ratio, extent) because they are easy to explain and they help
    to separate round objects from long objects.

    Returns a vector of 12 numbers. If no contour is found we return zeros,
    so the program never crashes.
    """
    height, width = gray.shape
    image_area = float(height * width)

    # --- 1. blur: kernel 5x5, sigma calculated automatically by OpenCV
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # --- 2. Canny edge detection.
    # 50 = lower threshold, 150 = upper threshold.
    # Pixels above 150 are certainly edges, pixels between 50 and 150 are
    # edges only if they are connected to a strong edge.
    edges = cv2.Canny(blurred, 50, 150)

    # --- 3. dilate the edges a little so broken lines become connected
    kernel = np.ones((3, 3), dtype=np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)

    # --- 4. find the contours.
    # RETR_EXTERNAL = only the outer contours (we do not need holes inside).
    # CHAIN_APPROX_SIMPLE = store only the corner points, not every pixel.
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        # No contour found (for example a completely flat image).
        # We return a vector of zeros instead of crashing.
        return np.zeros(12, dtype=np.float32)

    # --- 5. choose the contour that is really the main object
    main_contour = choose_main_contour(contours, image_area)

    if main_contour is None:
        return np.zeros(12, dtype=np.float32)

    area = cv2.contourArea(main_contour)
    perimeter = cv2.arcLength(main_contour, closed=True)

    if area <= 0 or perimeter <= 0:
        return np.zeros(12, dtype=np.float32)

    # --- 6. Hu moments of that contour
    moments = cv2.moments(main_contour)
    hu_moments = cv2.HuMoments(moments).flatten()   # 7 values

    # Log compression. We keep the sign, take log10 of the absolute value.
    # 1e-30 protects us against log10(0).
    hu_log = np.zeros(7, dtype=np.float32)
    for i in range(7):
        value = hu_moments[i]
        hu_log[i] = -np.sign(value) * np.log10(abs(value) + 1e-30)

    # After the logarithm the values are roughly between -30 and 30.
    # We divide by 30 and clip so that every Hu feature lies in [-1, 1].
    # Now all features have a similar size and none of them dominates
    # the distance calculation.
    hu_log = np.clip(hu_log / 30.0, -1.0, 1.0)

    # --- 5 extra geometric features, all already between 0 and 1 ---

    # how big is the object compared to the whole image
    area_ratio = area / image_area

    # how long is the boundary of the object.
    # Because the contour follows every small detail, the perimeter is usually
    # a few thousand pixels, so we divide by 16*width (= 4096 for a 256 image)
    # to bring this feature into the range [0, 1] like the others.
    perimeter_ratio = min(perimeter / (16.0 * width), 1.0)

    # circularity = 1 for a perfect circle, close to 0 for a long thin shape
    circularity = min((4.0 * np.pi * area) / (perimeter * perimeter), 1.0)

    # bounding box of the contour -> aspect ratio and extent
    x, y, w, h = cv2.boundingRect(main_contour)
    aspect_ratio = float(w) / float(h) if h > 0 else 0.0
    aspect_ratio = min(aspect_ratio / 3.0, 1.0)     # squeeze into [0, 1]

    # extent = how much of its bounding box the object really fills
    extent = area / float(w * h) if (w * h) > 0 else 0.0

    extra = np.array([area_ratio, perimeter_ratio, circularity,
                      aspect_ratio, extent], dtype=np.float32)

    shape_vector = np.concatenate([hu_log, extra]).astype(np.float32)
    return shape_vector


# =====================================================================
# 3. LOCAL FEATURE DESCRIPTOR - ORB
# =====================================================================

def extract_orb_features(gray):
    """
    Detect ORB keypoints and calculate their descriptors.

    WHAT ORB DOES
    -------------
    ORB = Oriented FAST and Rotated BRIEF.

    - FAST finds "corners": small places where the brightness changes in
      several directions (an eye, the tip of an ear, a spot on a leopard).
      Such points can be found again in another photo of the same object.
    - ORB also calculates the dominant orientation of every keypoint, so the
      descriptor still works if the object is rotated.
    - BRIEF then compares pairs of pixels around the keypoint and writes the
      result as bits. The descriptor is 32 bytes = 256 bits per keypoint.

    Because the descriptor is binary, two descriptors are compared with the
    HAMMING DISTANCE (simply: how many bits are different).

    Returns (keypoint_count, descriptors). 'descriptors' is None when the
    image has no detectable corners - we must handle that case everywhere.
    """
    keypoints, descriptors = orb_detector.detectAndCompute(gray, None)

    if descriptors is None:
        return 0, None

    return len(keypoints), descriptors


# =====================================================================
# All three descriptors together
# =====================================================================

def extract_hog_features(gray):
    """
    SHAPE DESCRIPTOR - HOG (Histogram of Oriented Gradients).

    THE IDEA
    --------
    The old shape descriptor had to pick ONE contour and hope it was the
    object. On a natural photograph that often fails (Canny also finds
    grass, clouds and furniture), which is why it was the weakest of the
    three descriptors.

    HOG does not extract any contour at all. It simply asks, for every small
    part of the image: "in which DIRECTIONS do the edges point here, and how
    strong are they?" The answer describes the local shape structure and
    does not depend on one fragile contour.

    THE STEPS
    ---------
    1. GRADIENTS. Sobel gives us how fast the brightness changes in the x
       direction (gx) and in the y direction (gy) for every pixel.

    2. MAGNITUDE AND ORIENTATION. From gx and gy we get, per pixel:

           magnitude = sqrt(gx^2 + gy^2)     how strong the edge is
           orientation = atan2(gy, gx)       which way the edge points

       The orientation is taken modulo 180 degrees ("unsigned"), because an
       edge from dark to light and the same edge from light to dark describe
       the same shape - only the direction of the line matters.

    3. CELLS. The image is cut into cells of HOG_CELL_SIZE pixels. With a
       256x256 image and 64-pixel cells that gives a 4 x 4 grid.

    4. HISTOGRAM PER CELL. Each cell gets a histogram with HOG_BINS = 9
       bins, one bin per 20 degrees (9 * 20 = 180). Every pixel votes into
       the bin of its orientation, and the vote is WEIGHTED BY ITS
       MAGNITUDE - so strong edges count more than weak noise. This
       weighting is what makes HOG describe real structure.

    5. ONE VECTOR. The 16 cell histograms are concatenated (4*4*9 = 144
       numbers) and the whole vector is L2-normalised, so it becomes a unit
       vector. That makes the descriptor independent of the overall contrast
       of the photo, and it is what lets distances.py use sqrt(2) as the
       largest possible distance.

    WHY NO "BLOCK NORMALISATION"?
    Textbook HOG adds a second normalisation over overlapping blocks of
    cells, to cancel illumination differences. We measured it and it made
    the results slightly WORSE here - because preprocessing already runs
    CLAHE, so the lighting is normalised before HOG ever sees the image.
    Leaving it out keeps the code simpler and performs better.

    Returns a 1D float32 vector of HOG_BINS * (IMAGE_SIZE / HOG_CELL_SIZE)^2
    numbers.
    """
    # --- 1. gradients in x and y with Sobel
    image = gray.astype(np.float32)
    gradient_x = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)

    # --- 2. how strong the edge is, and which way it points
    magnitude = cv2.magnitude(gradient_x, gradient_y)

    # np.arctan2 gives -180..180 degrees; "% 180" folds it to 0..180 so that
    # an edge and the same edge seen from the other side land in one bin
    orientation = np.rad2deg(np.arctan2(gradient_y, gradient_x)) % 180.0

    # which of the 9 bins each pixel belongs to
    degrees_per_bin = 180.0 / HOG_BINS
    bin_index = (orientation / degrees_per_bin).astype(np.int32)
    # a pixel at exactly 180 degrees would give bin 9, so clamp it
    bin_index = np.minimum(bin_index, HOG_BINS - 1)

    # --- 3 + 4. one magnitude-weighted histogram per cell
    cells_y = gray.shape[0] // HOG_CELL_SIZE
    cells_x = gray.shape[1] // HOG_CELL_SIZE

    histograms = []
    for row in range(cells_y):
        for column in range(cells_x):
            y_start = row * HOG_CELL_SIZE
            x_start = column * HOG_CELL_SIZE

            cell_bins = bin_index[y_start:y_start + HOG_CELL_SIZE,
                                  x_start:x_start + HOG_CELL_SIZE].ravel()
            cell_magnitude = magnitude[y_start:y_start + HOG_CELL_SIZE,
                                       x_start:x_start + HOG_CELL_SIZE].ravel()

            # np.bincount with 'weights' adds the magnitudes instead of
            # counting 1 per pixel - that is the magnitude weighting
            histogram = np.bincount(cell_bins, weights=cell_magnitude,
                                    minlength=HOG_BINS)
            histograms.append(histogram)

    vector = np.concatenate(histograms).astype(np.float32)

    # --- 5. L2 normalisation -> the vector has length 1
    # (1e-10 protects against a completely flat image where everything is 0)
    norm = float(np.linalg.norm(vector))
    if norm > 0:
        vector = vector / (norm + 1e-10)

    return vector.astype(np.float32)


def extract_main_roi(gray):
    """
    Cut out the REGION OF INTEREST (ROI) around the main object.

    WHY
    ---
    LBP and ORB look at the whole picture. If a dog stands on grass, most of
    the pixels are grass, so the texture descriptor mostly describes grass -
    and the dog can then match a completely unrelated picture that happens
    to have a similar background. Cutting the picture down to the object
    should reduce that.

    HOW (all of it is the contour logic we already have)
    ---------------------------------------------------
        1. blur -> Canny -> dilate -> findContours   (exactly like the
           shape descriptor does)
        2. choose_main_contour() picks the contour that is most probably
           the object
        3. cv2.boundingRect() gives the rectangle around that contour
        4. the rectangle is made 10% bigger on each side, so we do not cut
           the object off exactly at its edge
        5. the rectangle is clipped to the image, cut out, and resized back
           to IMAGE_SIZE x IMAGE_SIZE with the same "keep the aspect ratio
           and pad" method used in preprocessing

    WHEN WE DO NOT USE IT (the fallback)
    ------------------------------------
    Contour detection is not always reliable, so we simply return the full
    image when:
        * no contour was found at all
        * the region is smaller than ROI_MIN_SIDE pixels on a side
        * the region covers less than 5% of the image (it found a detail,
          not the object)
        * the region covers more than 90% of the image (cropping it would
          not remove any background anyway)

    This way the function can never return an empty or invalid crop, and
    the program never crashes because of it.
    """
    height, width = gray.shape
    image_area = float(height * width)

    # --- 1. the same edge + contour steps as the shape descriptor
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    kernel = np.ones((3, 3), dtype=np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    # --- 2. which contour is the main object
    main_contour = choose_main_contour(contours, image_area)
    if main_contour is None:
        return gray                      # fallback: no contour at all

    # --- 3. rectangle around it
    x, y, box_width, box_height = cv2.boundingRect(main_contour)

    # --- 4. make it ROI_MARGIN bigger on each side
    margin_x = int(round(box_width * ROI_MARGIN))
    margin_y = int(round(box_height * ROI_MARGIN))

    # clip to the image, so we never read outside the picture
    x_start = max(0, x - margin_x)
    y_start = max(0, y - margin_y)
    x_end = min(width, x + box_width + margin_x)
    y_end = min(height, y + box_height + margin_y)

    roi_width = x_end - x_start
    roi_height = y_end - y_start

    # --- the reliability checks (fallback to the full image)
    if roi_width < ROI_MIN_SIDE or roi_height < ROI_MIN_SIDE:
        return gray

    covered = (roi_width * roi_height) / image_area
    if covered < ROI_MIN_FRACTION or covered > ROI_MAX_FRACTION:
        return gray

    # --- 5. cut it out and bring it back to the normal processing size.
    # We keep the aspect ratio and pad, exactly like preprocessing does, so
    # the object is not stretched.
    roi = gray[y_start:y_end, x_start:x_end]

    scale = IMAGE_SIZE / float(max(roi_height, roi_width))
    new_width = max(1, int(round(roi_width * scale)))
    new_height = max(1, int(round(roi_height * scale)))
    resized = cv2.resize(roi, (new_width, new_height),
                         interpolation=cv2.INTER_AREA)

    pad_top = (IMAGE_SIZE - new_height) // 2
    pad_bottom = IMAGE_SIZE - new_height - pad_top
    pad_left = (IMAGE_SIZE - new_width) // 2
    pad_right = IMAGE_SIZE - new_width - pad_left

    return cv2.copyMakeBorder(resized, pad_top, pad_bottom,
                              pad_left, pad_right,
                              borderType=cv2.BORDER_CONSTANT, value=0)


def extract_sift_features(gray):
    """
    Detect SIFT keypoints and calculate their descriptors.

    WHAT SIFT DOES
    --------------
    SIFT = Scale-Invariant Feature Transform.

    Like ORB it looks for small, recognisable places in the image (corners,
    blobs, spots) that can be found again in another photo. The difference
    is HOW it does it:

    - SIFT searches for those points at MANY DIFFERENT SCALES. It blurs the
      image more and more and looks for points that stand out at every level
      of blur. So a keypoint found on a small photo of a car can still be
      found on a large photo of the same car - that is the "scale invariant"
      part of the name.
    - Each keypoint also gets a main orientation, so the descriptor still
      matches when the object is rotated.
    - The descriptor itself is a histogram of gradient directions in a 4x4
      grid around the keypoint, with 8 directions each: 4 * 4 * 8 = 128
      numbers.

    ORB vs SIFT, the practical difference:

        ORB  : 32 bytes of BITS   -> compared with the HAMMING distance
        SIFT : 128 FLOAT numbers  -> compared with the L2 (Euclidean) distance

    That is why distances.py needs a different matcher for each of them.

    Returns (keypoint_count, descriptors). 'descriptors' is None when the
    image has no detectable keypoints - the caller must handle that.
    """
    keypoints, descriptors = sift_detector.detectAndCompute(gray, None)

    if descriptors is None:
        return 0, None

    return len(keypoints), descriptors


def extract_all_features(gray):
    """
    Calculate the three descriptors of one preprocessed (grayscale) image
    and return them in a simple dictionary.

    If USE_ROI is switched on in config.py, the texture and the local
    features are taken from the region around the main object instead of
    the whole picture (see extract_main_roi). The shape descriptor always
    uses the full image, because it works on the main contour anyway.
    """
    # which image the texture and ORB features are calculated from
    if USE_ROI:
        texture_source = extract_main_roi(gray)
    else:
        texture_source = gray

    # which local descriptor is used (see LOCAL_DESCRIPTOR in config.py)
    if LOCAL_DESCRIPTOR == "sift":
        keypoint_count, local_descriptors = extract_sift_features(texture_source)
    else:
        keypoint_count, local_descriptors = extract_orb_features(texture_source)

    # which shape descriptor is used (see SHAPE_DESCRIPTOR in config.py)
    if SHAPE_DESCRIPTOR == "hog":
        shape_vector = extract_hog_features(gray)    # 144 numbers
    else:
        shape_vector = extract_shape_features(gray)  # 12 numbers (Hu moments)

    features = {
        "texture": extract_texture_features(texture_source),
        "shape": shape_vector,                       # always the full image
        "orb": local_descriptors,                    # ORB: N x 32, SIFT: N x 128
        "keypoints": keypoint_count,                 # how many keypoints
    }
    return features


def orb_keypoint_info(gray):
    """
    Used only by weighting.py.

    Returns (number_of_keypoints, average_strength_of_the_keypoints).

    Every ORB keypoint has a "response" value: it says how strong that corner
    is. A sharp, clear corner gets a high response, a weak corner found in a
    smooth area gets a low one. Looking at the average response tells us
    whether the keypoints of this image are really trustworthy or not.
    """
    keypoints = orb_detector.detect(gray, None)

    if keypoints is None or len(keypoints) == 0:
        return 0, 0.0

    average_response = float(np.mean([point.response for point in keypoints]))
    return len(keypoints), average_response

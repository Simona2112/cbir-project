# Content-Based Image Retrieval (CBIR)

Seminar / demo project for the course **Digital Image Processing**.

The program takes one query image and finds the **Top 10 visually most similar
images** in a dataset of 5 categories. The search uses only classical image
processing - **no machine learning, no neural networks, no colour**.

Built with **Python + OpenCV + NumPy** (Pillow and Tkinter only for the window).

> **The whole project works with DISTANCES.**
> A **small** distance means the images are **similar**.
> The results are sorted in **ascending** order and the 10 images with the
> **smallest** final distance are shown.

---

## 1. How to run it

```
pip install -r requirements.txt

python main.py              # normal start
python main.py --rebuild    # recalculate all features
python evaluate.py          # optional: measure how good the retrieval is
```

The first start calculates the features of all 200 dataset images (about 35
seconds) and saves them in `features/features.pkl`. Every later start just
loads that file.

If the feature file was written by an older version of the code, it is
detected and rebuilt automatically - you never have to delete it by hand.

## 2. Files

| File | What it does |
|---|---|
| `config.py` | all settings and all constants in one place (including `LBP_GRID`) |
| `preprocessing.py` | load image, resize + padding, grayscale, CLAHE |
| `feature_extractor.py` | the three descriptors: LBP, HOG, ORB |
| `distances.py` | the three **distance** measures + the weighted combination |
| `weighting.py` | calculates the automatic weights from the query image |
| `database.py` | builds the index, does the search, optional evaluation |
| `gui.py` | the Tkinter window (with a scrollable results area) |
| `main.py` | starts the program |
| `evaluate.py` | optional: Precision@10 measurement |

## 3. The pipeline

```
query image
    |
    v
PREPROCESSING   resize (keep aspect ratio) -> pad to 256x256
                -> grayscale -> CLAHE (fix the lighting)
    |
    +---------------------+---------------------+
    v                     v                     v
 TEXTURE                SHAPE                LOCAL
 spatial LBP         HOG                    ORB keypoints
 2x2 histograms      4x4 cells x 9 bins     + descriptors
 (1024 numbers)      (144 numbers)          (N x 32 bytes)
    |                     |                     |
    v                     v                     v
 CHI-SQUARE           EUCLIDEAN            HAMMING distance
 distance             distance             of the good matches
    |                     |                     |
    +---------------------+---------------------+
                          |
                  normalise each one
                  (divide by its own average)
                          |
                          v
       final_distance = w_texture * texture_distance
                      + w_shape   * shape_distance
                      + w_local   * local_distance

          (the weights come from the QUERY IMAGE itself)
                          |
                          v
              sort ASCENDING, take the Top 10
              (smallest distance = most similar)
```

The three descriptors are **never** glued together into one big feature
vector. Each one is compared on its own, with the distance measure that fits
its kind of data, and only the three resulting numbers are combined.

---

# EXPLANATIONS FOR THE PRESENTATION

## 3.1 Preprocessing

The dataset images are real photographs with very different sizes (up to
8192 x 5464 pixels). Feature vectors can only be compared if they come from
images of the same size, so every image is brought to **256 x 256**.

We do **not** simply stretch the image, because that would deform the object
and ruin the shape descriptor. Instead:

1. scale the image so its **longer side** becomes 256 pixels (aspect ratio stays)
2. put it in the middle of a black 256x256 canvas (`cv2.copyMakeBorder`)

Then it is converted to **grayscale** - this also guarantees that no colour
can influence the result, which is a requirement of the project.

Finally we apply **CLAHE** (Contrast Limited Adaptive Histogram Equalization),
which equalizes the histogram inside small 8x8 tiles and limits the contrast
so noise is not amplified. It makes the descriptors independent of how bright
the photo was taken. (Measured: it improves texture Precision@10 from 0.427 to
0.436 on this dataset.)

## 3.2 How LBP works (texture)

**LBP = Local Binary Pattern.**

For every pixel we look at its 8 neighbours and ask one question per
neighbour: *is the neighbour brighter than or equal to the centre?*

```
     neighbours                 example                 bit weights
     p0 p1 p2              120  140  95             1    2    4
     p7  C p3              110  [100] 90          128    C    8
     p6 p5 p4               80   105 130           64   32   16

     compare with centre 100:
     p0=120 -> 1    p1=140 -> 1    p2=95  -> 0    p3=90  -> 0
     p4=130 -> 1    p5=105 -> 1    p6=80  -> 0    p7=110 -> 1

     LBP code = 1 + 2 + 16 + 32 + 128 = 179
```

So every pixel gets a number between **0 and 255** describing the shape of
its local neighbourhood. LBP looks only at **differences between neighbours**,
so if the whole image gets brighter the codes stay the same - LBP is robust
against illumination changes.

**In the code:** instead of two slow Python loops over every pixel, we shift
the whole image with NumPy slicing (`padded[0:-2, 0:-2]` is the image of all
top-left neighbours) and compare all pixels at once.

### From a global histogram to SPATIAL LBP

The simplest version counts every LBP code of the whole image into **one**
histogram of 256 bins. That says *which* textures the image contains, but
never *where*: a stripe in the top-left corner and the same stripe in the
bottom-right corner land in exactly the same bin.

So the image is cut into a grid of `LBP_GRID x LBP_GRID` blocks, and every
block gets its own 256-bin histogram. The histograms are concatenated:

```
        +---------+---------+
        |  hist 0 |  hist 1 |      final vector =
        +---------+---------+      [ hist0 | hist1 | hist2 | hist3 ]
        |  hist 2 |  hist 3 |      4 * 256 = 1024 numbers
        +---------+---------+
```

Two details that matter:

1. The LBP is calculated **once on the complete image** and only then cut
   into blocks. If the LBP were run separately on each block, the pixels on
   the block borders would have no real neighbours and OpenCV would invent
   them by copying - fake texture along every block edge.

2. Each block histogram is divided by the pixel count of **that block**, so
   each sums to 1.0 and the whole vector sums to *the number of blocks*. The
   chi-square distance divides by the number of blocks to compensate (see
   section 4.1).

### Which grid size? (a measured result worth presenting)

`LBP_GRID` in `config.py` is a single number. We measured every setting on
this dataset (texture-only Precision@10, random baseline 0.20):

| grid | feature length | texture-only P@10 | combined P@10 | P@1 |
|---|---|---|---|---|
| 1x1 (global) | 256 | 0.436 | 0.385 | 0.500 |
| **2x2 (used)** | **1024** | **0.439** | **0.387** | **0.500** |
| 4x4 | 4096 | 0.414 | 0.341 | 0.430 |
| 8x8 | 16384 | 0.413 | - | - |

**A finer grid makes the system worse**, which is a genuinely interesting
result to explain:

* Spatial LBP comes from **face recognition**, where every image is aligned -
  the eyes, nose and mouth are always in the same block. Then the grid is a
  huge help.
* Our photos are **not aligned**. A cat filling the frame and a cat sitting in
  the bottom-left corner produce completely different block histograms, so a
  fine grid mostly adds *position noise* instead of useful structure.
* A fine grid also leaves **few pixels per bin** (at 4x4, about 16 counts per
  bin), and chi-square divides by `(a + b)`, so nearly empty bins become
  noisy.
* The one category a grid helps is **flowers** (0.36 -> 0.40), because flower
  photographs are usually centred close-ups - they are effectively aligned.

2x2 is the compromise that is used: it adds coarse "upper/lower, left/right"
information and is the only setting that beats the global histogram.

## 3.3 How the shape descriptor works - HOG

**HOG = Histogram of Oriented Gradients.**

The shape descriptor does **not** extract a contour. Instead it asks, for
every small part of the image: *in which directions do the edges point here,
and how strong are they?*

```
grayscale -> Sobel gradients -> magnitude + orientation
          -> 4x4 grid of cells -> 9-bin histogram per cell
          -> concatenate -> L2 normalise -> 144 numbers
```

1. **Gradients.** `cv2.Sobel` gives how fast the brightness changes in the
   x direction (`gx`) and the y direction (`gy`) for every pixel.

2. **Magnitude and orientation.** From those two numbers:

   ```
   magnitude   = sqrt(gx^2 + gy^2)     how strong the edge is
   orientation = atan2(gy, gx)         which way it points
   ```

   The orientation is folded to **0-180 degrees** ("unsigned"), because an
   edge from dark to light and the same edge from light to dark describe the
   same shape - only the direction of the line matters.

3. **Cells.** The 256x256 image is cut into cells of 64x64 pixels, giving a
   **4 x 4 grid**.

4. **One histogram per cell.** Each cell gets **9 bins**, one per 20 degrees
   (9 x 20 = 180). Every pixel votes into the bin of its orientation, and the
   vote is **weighted by its magnitude** - so strong edges count and weak
   noise does not. This weighting is what makes HOG describe real structure.

5. **One vector.** The 16 cell histograms are concatenated (4 x 4 x 9 = 144
   numbers) and the whole thing is **L2-normalised** into a unit vector, so
   the descriptor does not depend on the overall contrast of the photo.

### Why HOG replaced Hu moments

The previous descriptor had to choose **one** contour and hope it was the
object. On a natural photograph Canny also finds grass, clouds and furniture,
so the "main contour" was often the background - which is why shape was the
weakest of the three descriptors. HOG needs no contour at all: it describes
the distribution of edge directions in local regions, so there is nothing
fragile to go wrong.

Measured on this dataset (shape-only Precision@10, random baseline 0.20):

| descriptor | numbers | shape-only P@10 | combined P@10 | P@1 |
|---|---|---|---|---|
| Canny + contour + Hu moments | 12 | 0.268 | 0.387 | 0.500 |
| **HOG** | **144** | **0.449** | **0.482** | **0.530** |

Paired over the 200 queries the improvement is **+0.153 +/- 0.023** for
shape alone and **+0.079 +/- 0.015** combined - many times bigger than the
measurement noise, so this is a real improvement and not luck.

The old Hu descriptor is still in the code. `SHAPE_DESCRIPTOR = "hu"` in
`config.py` switches back to it, and the feature index rebuilds itself.

### Choosing the cell size

`HOG_CELL_SIZE` was measured, not guessed (shape-only Precision@10):

| cell size | grid | numbers | P@10 |
|---|---|---|---|
| 16 px | 16x16 | 2304 | 0.390 |
| 32 px | 8x8 | 576 | 0.420 |
| **64 px** | **4x4** | **144** | **0.449** |
| 128 px | 2x2 | 36 | 0.386 |

It improves up to 64 and gets worse again at 128, so 64 is a real optimum
rather than "coarser is always better". The same ordering held on two random
halves of the dataset checked separately.

Textbook HOG also adds a second normalisation over overlapping **blocks** of
cells. We tested it and it made the results slightly *worse* here, because
block normalisation exists to cancel illumination differences and the
preprocessing already does that with CLAHE. Leaving it out is both simpler
and better.

## 3.4 How ORB works (local features)

**ORB = Oriented FAST and Rotated BRIEF.**

* **FAST** finds *corners*: places where the brightness changes in several
  directions - a headlight, an eye, the tip of a petal.
* **Oriented**: ORB computes the dominant orientation of each keypoint, so the
  descriptor still works when the object is rotated.
* **BRIEF** describes each keypoint by comparing pairs of pixels around it and
  writing the results as bits: **32 bytes = 256 bits** per keypoint.

Because the descriptor is binary, two descriptors are compared with the
**Hamming distance** = how many of the 256 bits are different. That is why the
matcher is created with `cv2.NORM_HAMMING`.

---

## 4. The three DISTANCES

Every distance is **0.0 for identical** and **1.0 for completely different**.

### 4.1 Texture distance - Chi-square

```
d = 0.5 * sum over all bins of  (h1[i] - h2[i])^2 / (h1[i] + h2[i])

    then divided by the number of blocks (4 for a 2x2 grid)
```

**Why chi-square and not a plain difference?** In an LBP histogram a few bins
are very large (the codes that appear in flat areas) and most bins are small.
A plain difference would only notice the few big bins. Chi-square divides each
bin's difference by how big that bin is, so a difference of 0.01 in a small
bin counts as much as in a large bin - and the small bins are exactly the
interesting texture patterns.

**Range, and why we divide by the number of blocks:** for *one* histogram
that sums to 1.0 this formula is already inside [0, 1] - identical histograms
give 0.0, and two histograms with no bin in common give 0.5 + 0.5 = 1.0.

But the descriptor is a **spatial** LBP: 4 block histograms glued together,
each normalised on its own, so the whole vector sums to **4**. Without a
correction the distance could reach 4.0 and everything would clip to 1.0,
destroying the ranking. So the sum is divided by the number of blocks - which
is exactly the same as computing the chi-square distance of each block
separately and taking the **average** of the 4 results.

### 4.2 Shape distance - Euclidean

```
d = sqrt( sum( (a[i] - b[i])^2 ) )      then divided by sqrt(2)
```

The ordinary straight-line distance between the two HOG vectors. It is
**already** a distance, so - unlike an earlier version of this project - we
do **not** convert it with `1 / (1 + d)`.

**Why exactly sqrt(2)?** The HOG vectors are L2-normalised, so both are unit
vectors, and every HOG value is >= 0. Two non-negative unit vectors can at
worst stand at right angles to each other, and then their distance is exactly
sqrt(2). That is the largest value that can ever occur, so dividing by
sqrt(2) is not a tuned constant - it follows from the mathematics. Measured
range on this dataset: 0.286 to 0.921.

### 4.3 Local distance - Hamming distance of the good matches

1. `knnMatch(k=2)` finds, for every descriptor of the query, its two closest
   descriptors in the other image.
2. **Lowe's ratio test** keeps a match only if `best < 0.75 * second_best`.
   If the two candidates are almost equally good, the keypoint looks like many
   other points and cannot be trusted.
3. **Quality** = the mean Hamming distance of the good matches, divided by 256.
4. **Confidence** = `min(good_matches / 10, 1.0)`.
5. The distance blends the two:

```
local_distance = confidence * quality + (1 - confidence) * 1.0
```

**Why the confidence part?** The mean is only meaningful if it was computed
from enough matches. With a single lucky match the mean can be very small by
pure chance and that image would jump to first place. With 10 or more good
matches we trust the measurement fully; with 2 matches we trust it only 20%
and the distance stays close to 1.0 ("we did not really find this object
again"). If there are no descriptors or no good matches at all, the function
returns the worst distance 1.0 instead of crashing.

**What was wrong before:** the old version returned
`good_matches / number_of_descriptors`, which was about **0.005** for almost
every pair of images. The local score was therefore practically always 0 and
carried no information at all.

## 5. Making the three distances comparable

This step is **essential**, and it is worth explaining to the professor.

All three distances are mathematically inside [0, 1], but in practice they use
very different parts of that range. Measured on this dataset:

| distance | typical (median) value |
|---|---|
| texture (chi-square) | **0.05** |
| shape (Euclidean) | **0.53** |
| local (ORB) | **0.84** |

If those numbers went straight into the weighted average, shape and local
would decide the ranking **no matter what the weights say**, simply because
their numbers are about 16 times bigger. The texture weight could be 90% and
it would still barely change the result.

So for each query we divide every distance by the **average** distance of that
descriptor to all database images:

```
normalised[i] = distance[i] / mean(all distances of this descriptor)
```

After this, 1.0 means "an average image", below 1.0 means "closer than
average". All three descriptors now speak the same language and the weights
really control the balance.

Measured effect (Precision@10 with equal weights): **0.296 without
normalisation, 0.382 with it.**

## 6. How the automatic weights are calculated

The weights are **not** fixed and the user does **not** choose them - they are
computed from the **query image only**.

| weight | what we measure | why |
|---|---|---|
| **texture** | standard deviation of the **Laplacian** | the Laplacian reacts to fast brightness changes (fur, petals, gravel). Smooth image -> small value, detailed image -> large value |
| **shape** | 50% **main contour area** + 40% **solidity** + 10% **edge density** | see "the shape importance" below |
| **local** | (50% **keypoint count** + 50% **average keypoint strength**) **x LOCAL_TRUST** | see below |

Each measurement is divided by a reference value from `config.py` (roughly the
90th percentile measured on this dataset), a small minimum of 0.05 is added so
no descriptor can vanish, and then the three values are normalised:

```
total = texture_importance + shape_importance + local_importance

texture_weight = texture_importance / total
shape_weight   = shape_importance   / total
local_weight   = local_importance   / total

texture_weight + shape_weight + local_weight = 1.0
```

### The shape importance (why solidity, not edge density)

The first version of this heuristic used `0.5 * edge_density + 0.5 * area`.
Edge density is the percentage of pixels Canny marks as an edge - but on a
cat, a dog or a flower, the fur, petals and background texture create a huge
number of Canny edges. So the **same visual detail raised the texture weight
and the shape weight at the same time**: the two weights were measuring the
same thing twice.

Measured on this dataset, the correlation between shape importance and
texture importance was **+0.62** (edge density alone: +0.70).

The shape weight should instead answer: *does this image contain one clear,
dominant silhouette?* The new formula is

```
shape_importance = 0.50 * area_score      (is there a big object?)
                 + 0.40 * solidity        (is its outline clean or ragged?)
                 + 0.10 * edge_score      (small extra contribution)
```

**Solidity** = `contourArea / convexHullArea`. The convex hull is the smallest
convex shape around the contour - imagine stretching a rubber band around the
object. A compact object (an apple, a car body) fills its rubber band almost
completely; a ragged outline full of deep notches - exactly what fur and
petals produce - leaves a lot of empty space. It is already a number between
0 and 1, so it needs no reference constant.

Average solidity per category:

| fruits | flowers | cars | dogs | cats |
|---|---|---|---|---|
| 0.82 | 0.72 | 0.67 | 0.64 | **0.53** |

The new correlation with texture importance is **+0.53** instead of +0.62. It
is not zero, because the area score itself grows a little on detailed images -
but object size is the legitimate core of "is there a silhouette here", so we
keep it.

**A measure that did NOT work.** The obvious idea is "how dominant is the main
contour compared to the other contours". It fails here: after the Canny edges
are dilated, almost everything melts into ONE contour (the median number of
valid contours per image is 1), so the measure was 1.0 for practically every
image. Computing the contours without dilation does not help either - then
nearly all of them fall below the 1% size filter.

**What it does in practice.** Within a category the shape weight now follows
how clean the outline is:

| image | solidity | shape weight (old -> new) |
|---|---|---|
| `car_016.jpg` | 0.91 | 0.45 -> **0.47** |
| `flower_019.jpg` | 0.91 | 0.44 -> **0.48** |
| `cat_031.jpg` | 0.34 | 0.35 -> **0.33** |
| `fruit_034.jpg` | 0.50 | 0.35 -> **0.32** |

### The LOCAL_TRUST factor (this fixes a real problem)

ORB almost always returns the maximum number of keypoints - on this dataset
the **median is 285 out of 300**. So a weight based on the keypoint count is
practically constant and high, and the old heuristic gave local features
**45-50% of the total weight for nearly every query**.

That is wrong, because a high keypoint count only says the *query* has good
corners. It cannot say whether any database image contains the **same object**
- and that is the only situation where ORB really works. Measured on its own,
ORB is the weakest of the three descriptors here (P@10 0.264 vs 0.436 for
texture).

So the local importance is multiplied by `LOCAL_TRUST = 0.4`. The local weight
still varies from query to query, but it can no longer dominate:

| | before | now |
|---|---|---|
| local weight range | 0.26 - 0.62 | **0.12 - 0.40** |
| local weight median | ~0.36 | **0.20** |

## 7. Final distance and the Top 10

```
final_distance = texture_weight * texture_distance
               + shape_weight   * shape_distance
               + local_weight   * local_distance
```

The list is sorted **ascending** (`results.sort(key=...)` with no `reverse`)
and the **first 10** entries are shown. **Lower = more similar.**

### The query image is NOT removed

If you pick an image that is also inside the dataset, it appears as result #1
with a distance of **0.000**. This is **intentional** and is a good way to
prove during the demo that the pipeline is correct. The search function never
filters anything out.

(The only exception is `evaluate.py`: for an honest measurement it skips that
one self-match, because otherwise every query would get one correct hit for
free. That happens only in the evaluation script, never in the search itself.)

## 8. Why the folder / category names are NOT used

The dataset is organised in folders (`cars`, `cats`, `dogs`, `flowers`,
`fruits`) only because the files have to be stored somewhere. During the
search the program:

* scans **all** subfolders with `os.walk` and puts every image into **one flat
  list** - there is no separate list per category;
* compares the query with **every** entry of that list, so a cat query is
  compared with all 200 indexed images, not only with the ones in `cats/`;
* computes the distances **only** from the LBP histogram, the shape vector and
  the ORB descriptors. The label is stored in the index but it is never read by
  `texture_distance`, `shape_distance`, `local_distance` or `search`.

The label is read in exactly two places, and **both happen after the ranking is
already finished**: to show the category under each thumbnail, and to compute
Precision@10. Neither can change the order of the results.

## 9. Console debug output

Every search prints its full calculation in the console, so you can verify the
ranking during the presentation:

```
==============================================================================
QUERY
------------------------------------------------------------------------------
  ORB keypoints detected : 272
  raw importance  texture: 0.669   shape: 0.587   local: 0.224
  normalised weights     : texture 44.1%   shape 39.1%   local 16.8%
  (the three weights add up to 1.000)

TOP 10 RESULTS - sorted by final distance, SMALLEST first
------------------------------------------------------------------------------
  rank final     texture   shape     local      image
  1    0.0000    0.0000    0.0000    0.0000     dataset/cats/cat_001.jpg
  2    0.4680    0.0240    0.1732    0.5965     dataset/cats/cat_034.jpg
  3    0.4685    0.0125    0.3747    0.2684     dataset/cats/cat_014.jpg
  ...
==============================================================================
```

---

## 10. Results measured on this dataset

`python evaluate.py` - every one of the 200 images used as a query, against
the 200 indexed images from 5 categories (the self-match is skipped):

| method | Precision@10 |
|---|---|
| random guessing (baseline) | 0.200 |
| local features (ORB) only | 0.264 |
| shape (HOG) only | 0.449 |
| texture (spatial LBP) only | 0.439 |
| **combined, dynamic weights** | **0.482** |

**Precision@1 of the combined system is 0.530** - for half of the queries the
first returned image (after the query itself) is the right category, against a
random baseline of 0.20.

Per category (combined system):

| category | Precision@10 |
|---|---|
| cars | 0.73 |
| flowers | 0.58 |
| fruits | 0.48 |
| dogs | 0.34 |
| cats | 0.27 |

### How to discuss this with the professor

The combined system is clearly much better than random, but on this dataset
the texture descriptor alone is the strongest single feature, and mixing it
with the two weaker descriptors pulls the average down a little. This is an
honest and normal result for classical CBIR:

* **Cars work best (0.60)** because they have smooth metal surfaces and strong
  straight edges - both the texture and the shape descriptor agree on them.
* **Dogs work worst (0.23)** because dogs and cats have almost the same fur
  texture and a similar silhouette; without colour they are very hard to tell
  apart. The 2x2 spatial grid was added partly to separate them and it barely
  helped (0.21 -> 0.23) - the two classes really are close in grayscale.
* **Shape is weak on natural photographs.** Even with the improved contour
  selection, Canny still finds grass, leaves and furniture, so the "main
  contour" is not always the object.
* **ORB is a near-duplicate detector.** It is excellent when the *same* object
  appears in two photographs, but two *different* cats are different objects.
  That is exactly why the confidence term pushes its distance towards 1.0 when
  only a couple of matches are found.

## 11. Error handling

The program does not crash when something goes wrong:

| problem | what happens |
|---|---|
| broken or non-image file | `load_image` returns `None`, the image is skipped |
| image with no contours | the shape vector is all zeros |
| image with no ORB keypoints | the local distance is the worst value 1.0 |
| no good ORB matches | the local distance is the worst value 1.0 |
| query image cannot be opened | the GUI shows an error message box |
| query image is nearly black | the GUI shows a warning but still searches |
| empty dataset folder | a clear message is printed, the GUI does not open |
| damaged or outdated `features.pkl` | it is detected and the index is rebuilt automatically |

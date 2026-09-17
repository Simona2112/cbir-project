# Content-Based Image Retrieval (CBIR)

Seminar project for the course **Digital Image Processing**.

The program takes one query image and returns the **10 most visually similar
images** from a small dataset. It is a *retrieval* system, not a classifier —
it never predicts a category, it only ranks the dataset images by how similar
they look to the query.

Everything is done with classical image processing: **Python + OpenCV + NumPy**,
with Tkinter for the window and Pillow for showing the thumbnails.
No machine learning, no neural networks, no pretrained models, and no colour
features (all descriptors work on the grayscale image).

> The whole project works with **distances**.
> A **smaller** distance means the images are **more similar**.
> Results are sorted in ascending order.

---

## 1. Installation and running

```bash
pip install -r requirements.txt

python main.py              # start the GUI
python main.py --rebuild    # force the feature index to be recalculated
python evaluate.py          # run the Precision@10 evaluation
```

`requirements.txt` needs `opencv-python`, `numpy` and `Pillow`. Tkinter is part
of the standard library on Windows and macOS; on Linux it may need
`sudo apt install python3-tk`.

The first start extracts the features of all 200 dataset images (about a
minute) and saves them in `features/features.pkl`. Every later start just loads
that file, and a search then takes around a tenth of a second.

**The index rebuilds itself when it needs to.** `features.pkl` stores a version
number and the descriptor settings it was built with (image size, LBP grid,
LBP scales, ROI on/off, shape descriptor, local descriptor). If any of those
change in `config.py`, the file is detected as outdated and rebuilt
automatically — you never have to delete it by hand. `--rebuild` forces it
anyway.

## 2. Dataset

200 images in 5 categories, 40 images each:

```
dataset/cars/  cats/  dogs/  flowers/  fruits/
```

The folder names are used **only** for displaying the category under a result
and for the evaluation. They are never used during the search — see section 7.

## 3. Project structure

```
ContentBasedImageRetrieval/
│
├── dataset/
│   ├── cars/          40 images
│   ├── cats/          40 images
│   ├── dogs/          40 images
│   ├── flowers/       40 images
│   └── fruits/        40 images
│
├── features/
│   └── features.pkl       saved feature index
│
├── query images/          a few test images that are NOT in the dataset
│
├── config.py              all settings and constants
├── preprocessing.py       resize, padding, grayscale, contrast check, CLAHE
├── feature_extractor.py   LBP, HOG and ORB descriptors
├── distances.py           the three distances and the final combination
├── weighting.py           automatic weights calculated from the query
├── database.py            index building, saving/loading, search
├── gui.py                 the Tkinter window
├── main.py                starts the application
├── evaluate.py            Precision@10 evaluation
├── requirements.txt
└── README.md
```

## 4. How the system works

```
query image
    |
PREPROCESSING   resize to 256x256 (aspect ratio kept, black padding)
                -> grayscale -> contrast check -> CLAHE
    |
    +--------------------+--------------------+
    v                    v                    v
 TEXTURE              SHAPE                LOCAL
 spatial LBP          HOG                  ORB
 2x2 grid             4x4 cells x 9 bins   up to 300 keypoints
 1024 values          144 values           32-byte descriptors
    |                    |                    |
    v                    v                    v
 Chi-square           Euclidean            Hamming matching
 distance             distance             + Lowe ratio test
    |                    |                    |
    +--------------------+--------------------+
                         |
             each distance normalised separately
             (divided by its own average over the dataset)
                         |
    final = w_texture * texture + w_shape * shape + w_local * local
             (the weights come from the QUERY IMAGE itself)
                         |
             sort ascending -> Top 10
```

The three descriptors are compared **separately**, each with the distance
measure that fits it. They are **never concatenated** into one big feature
vector — only the three resulting numbers are combined at the end.

### 4.1 Preprocessing

Every image goes through the same steps so the descriptors are comparable:

1. **Resize to 256x256 keeping the aspect ratio.** The longer side is scaled to
   256 and the rest is filled with **black padding**, so the object is not
   stretched.
2. **Convert to grayscale.** This also guarantees that no colour information
   can influence the results.
3. **Contrast check.** The standard deviation of the grayscale image is
   measured *before* CLAHE. If it is below `MIN_CONTRAST_STD` (12.0) the image
   is marked as unusable — such an image carries almost no information and is
   skipped when building the index.
4. **CLAHE** (Contrast Limited Adaptive Histogram Equalization, 8x8 tiles,
   clip limit 2.0) equalises the local contrast, so a darker and a brighter
   photo of the same object give similar features.

### 4.2 Texture — Spatial LBP

For every pixel, LBP compares the 8 neighbours at **radius 1** with the centre
pixel. Each comparison gives one bit ("is the neighbour brighter or equal?"),
and the 8 bits form a number between 0 and 255 that describes the local
texture pattern. Because LBP only looks at *differences* between neighbours, it
is not affected by the overall brightness of the image.

To keep some spatial information the image is split into a **2x2 grid**. Every
block gets its own 256-bin histogram, normalised so it sums to 1, and the four
histograms are concatenated:

```
4 blocks x 256 bins = 1024 values
```

The LBP image is calculated **once on the whole image** and only then split
into blocks, so the pixels on the block borders still use their real
neighbours.

### 4.3 Shape — HOG

HOG (Histogram of Oriented Gradients) describes the distribution of edge
directions in local regions. Unlike a contour-based descriptor it does not
depend on extracting one single contour correctly, which is what makes it more
stable on natural photographs.

1. **Sobel** (3x3) gives the gradients `gx` and `gy`.
2. For every pixel: `magnitude = sqrt(gx² + gy²)` and
   `orientation = atan2(gy, gx)`, folded into **0–180 degrees** (unsigned — an
   edge and the same edge from the other side describe the same shape).
3. The image is divided into **64x64 pixel cells**, which gives a **4x4 grid**.
4. Each cell gets a **9-bin** histogram (one bin per 20 degrees). Every pixel
   votes into the bin of its orientation, **weighted by its magnitude**, so
   strong edges count more than weak noise.
5. The 16 histograms are concatenated and the whole vector is **L2 normalised**:

```
4 x 4 cells x 9 bins = 144 values
```

### 4.4 Local features — ORB

ORB (Oriented FAST and Rotated BRIEF) finds up to **300 keypoints** — small
recognisable places such as corners — and describes each one with a 32-byte
(256-bit) binary descriptor. Because the descriptors are binary they are
compared with the **Hamming distance**.

## 5. The three distances

Every distance is **0.0 for identical** and **1.0 for completely different**.

### Texture — Chi-square

```
d = 0.5 * Σ (h1[i] - h2[i])² / (h1[i] + h2[i])      then divided by 4 blocks
```

Chi-square divides each bin's difference by how large that bin is, so the many
small bins (which carry the interesting texture patterns) count as much as the
few large ones. Since each block histogram sums to 1, the concatenated vector
sums to 4, so the result is divided by the number of blocks to bring it back
into [0, 1] — which is the same as averaging the four per-block distances.

### Shape — Euclidean

```
d = sqrt( Σ (a[i] - b[i])² )      then divided by sqrt(2)
```

The HOG vectors are L2 normalised and all their values are ≥ 0, so two of them
can at worst stand at right angles to each other, and then their distance is
exactly `sqrt(2)`. That is the largest value that can occur, so dividing by it
is not a tuned constant — it follows from the descriptor.

### Local — ORB matching

1. `BFMatcher(cv2.NORM_HAMMING)` with `knnMatch(k=2)` finds the two closest
   descriptors for each query descriptor.
2. **Lowe's ratio test** (`0.75`) keeps only matches where the best candidate is
   clearly better than the second best; ambiguous matches are thrown away.
3. The distance combines how *good* the matches are with how *many* there are:

```
quality    = mean Hamming distance of the good matches / 256
confidence = min(good matches / 10, 1.0)

local_distance = confidence * quality + (1 - confidence) * 1.0
```

With 10 or more good matches the measured quality is trusted fully. With only
one or two matches the mean could be small by pure luck, so the distance stays
close to 1.0. If there are no descriptors or no good matches at all, the
function returns 1.0 instead of failing.

## 6. Dynamic weights and the final distance

The weights are **not fixed** and the user does not choose them — they are
calculated from the **query image only**, in `weighting.py`:

| weight | what is measured |
|---|---|
| **texture** | standard deviation of the **Laplacian** — how much fine detail the image has |
| **shape** | `0.50 ×` main-contour area `+ 0.40 ×` its solidity `+ 0.10 ×` Canny edge density |
| **local** | (`0.5 ×` ORB keypoint count `+ 0.5 ×` average keypoint response) `× LOCAL_TRUST (0.4)` |

Each measurement is divided by a reference value from `config.py`, a minimum of
`0.05` is added so no descriptor can disappear completely, and the three values
are normalised so that:

```
texture_weight + shape_weight + local_weight = 1.0
```

`LOCAL_TRUST = 0.4` limits how much weight ORB can ever receive. ORB almost
always returns close to the maximum number of keypoints, so without this limit
local features took a large share of the weight even though they are the
weakest descriptor here.

**Before combining, each of the three distance lists is normalised separately**
by dividing it by its own average over the whole dataset. This matters: the raw
distances live on very different scales, and without this step the descriptor
with the largest raw numbers would decide the ranking no matter what the
weights say. After normalisation, 1.0 means "an average image" for every
descriptor, so the weights really control the balance.

```
final_distance = w_texture * texture + w_shape * shape + w_local * local
```

The list is then sorted **ascending** and the first 10 entries are the result.

## 7. Why the category folders are not used for retrieval

During the search the program puts every dataset image into **one flat list**
and compares the query with **all 200 of them** — there is no separate list per
category. The distances are computed only from the LBP histogram, the HOG
vector and the ORB descriptors. The label is stored in the index but it is
never read by `texture_distance`, `shape_distance`, `local_distance` or
`search`.

The label is used in exactly two places, and **both happen after the ranking is
finished**: to print the category under a thumbnail, and to calculate
Precision@10.

## 8. The GUI

`python main.py` opens a simple Tkinter window:

- **Choose Query Image** — pick any image from disk (also one that is not in the
  dataset).
- A preview of the query and the automatically calculated weights, shown as
  percentages.
- **Search Similar Images** — runs the search.
- The **Top 10 results** in a scrollable area (Canvas + scrollbar, the mouse
  wheel works), each card showing the rank, the thumbnail, the category folder,
  the **final distance** and the three separate distances `T:`, `S:`, `L:`.
- A reminder under the header that **lower distance means more similar**.
- If the query image comes from a dataset folder, the status line also shows
  Precision@10 for that search.

Every search also prints its full calculation to the console: the ORB keypoint
count, the raw importance values, the normalised weights, and the Top 10 with
all four distances. This is useful for checking that the ranking is correct.

**Note:** the query image is deliberately **not** removed from the results. If
you pick an image that is inside the dataset, it appears as result #1 with a
distance of 0.000. That is intended — it is a quick way to show that the
pipeline works.

## 9. Evaluation

```bash
python evaluate.py
```

`evaluate.py` uses **every indexed image as a query**, and — unlike the GUI —
**removes the query image itself** from its own result list before counting, so
each query does not get one correct hit for free.

Results on this dataset (200 images, 5 categories):

| method | Precision@10 |
|---|---|
| random baseline | 0.200 |
| local features (ORB) only | 0.264 |
| texture (spatial LBP) only | 0.439 |
| shape (HOG) only | 0.449 |
| **combined, dynamic weights** | **0.482** |

**Combined Precision@1 = 0.530** — for about half of the queries the first
returned image is from the correct category.

Per category (combined system):

| category | Precision@10 |
|---|---|
| cars | 0.73 |
| flowers | 0.58 |
| fruits | 0.48 |
| dogs | 0.34 |
| cats | 0.27 |

The combined system is clearly better than any single descriptor and more than
twice as good as random guessing.

## 10. Limitations

- **Cats and dogs are the weakest categories** (0.27 and 0.34). In grayscale
  they have very similar fur texture and similar silhouettes, so the
  descriptors cannot separate them well. Colour would be the most useful cue
  here, and this project deliberately does not use it.
- **ORB is essentially a near-duplicate detector.** It works well when the
  *same* object appears in two photos, but two different cats are two different
  objects. On this dataset most image pairs produce only a couple of good
  matches, which is why the local distance is often close to 1.0 and why ORB is
  the weakest of the three descriptors (0.264).
- **HOG is not rotation invariant** and assumes the object is roughly in the
  same part of the image, since the cells are at fixed positions. The same is
  true for the 2x2 LBP grid.
- **The dataset is small** (200 images, 40 per category), so single numbers in
  the evaluation move easily — a difference of one or two images already
  changes Precision@10 noticeably.
- **The search is a linear scan.** Every query is compared with all 200 images.
  That is fast enough here (about 0.1 s), but the cost grows linearly, so it
  would not scale to thousands of images without an indexing structure.
- **The shape weight and the shape descriptor measure different things.** The
  weight in `weighting.py` is still derived from contour area and solidity,
  while the descriptor itself is HOG. It works, but aligning them would be a
  sensible improvement.

## 11. Optional code from earlier experiments

While developing the project several alternatives were implemented and
compared. They are **not used in the final version**, but the code is still
there and can be switched on in `config.py`:

| setting | final value | alternative still in the code |
|---|---|---|
| `SHAPE_DESCRIPTOR` | `"hog"` | `"hu"` — the older Canny + contour + Hu moments descriptor (`extract_shape_features`) |
| `LOCAL_DESCRIPTOR` | `"orb"` | `"sift"` — SIFT with L2 matching (`extract_sift_features`) |
| `USE_ROI` | `False` | `True` — crop a region of interest around the main contour before extracting texture and local features (`extract_main_roi`) |
| `LBP_SCALES` | `[1]` | e.g. `[1, 2]` — multi-scale LBP, a second radius added as extra histograms |

Changing any of these rebuilds the feature index automatically. They were kept
so the comparisons can be repeated, not because they improved the results —
HOG was the only one of these changes that was measurably better than what it
replaced.

## 12. Error handling

The program does not crash on a single bad file:

| problem | what happens |
|---|---|
| broken or non-image file | `load_image` returns `None` and the image is skipped |
| image with too little contrast | marked unusable and left out of the index |
| no contours found | the Hu descriptor returns a zero vector (HOG does not need contours) |
| no ORB keypoints / no good matches | the local distance is the worst value, 1.0 |
| query image cannot be opened | the GUI shows an error message box |
| query image has low contrast | the GUI shows a warning but still searches |
| empty dataset folder | a clear message is printed and the GUI does not open |
| outdated or damaged `features.pkl` | detected and rebuilt automatically |

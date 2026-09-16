"""
database.py
-----------
Builds and stores the FEATURE DATABASE (the "index") of the dataset,
and performs the actual search.

Why an index?
Calculating LBP, contours and ORB for ~600 images takes some time. We do it
only once and save the result in features/features.pkl with pickle. Every
later search just loads that file, so searching is fast.

VERY IMPORTANT RULE OF THE PROJECT
----------------------------------
The folder name (cars, cats, dogs, ...) is stored ONLY so that we can show
the file path and calculate Precision@10 AFTER the search. It is never used
inside the distance calculation. The search always compares the query with
EVERY image of EVERY folder.
"""

import os
import pickle

from config import (DATASET_DIR, FEATURES_DIR, FEATURES_FILE,
                    VALID_EXTENSIONS, MAX_IMAGES_PER_CLASS, TOP_K,
                    FEATURES_FORMAT_VERSION, IMAGE_SIZE, LBP_GRID, LBP_SCALES,
                    USE_ROI, SHAPE_DESCRIPTOR, LOCAL_DESCRIPTOR)
from preprocessing import load_and_preprocess
from feature_extractor import extract_all_features
from distances import (texture_distance, shape_distance,
                       local_distance, combine_distances)
from weighting import compute_weights


def list_dataset_images():
    """
    Walk through the dataset folder and collect the paths of all images.

    os.walk goes through every subfolder automatically (recursively), so we
    never have to write any file name in the code.

    We also remember the name of the folder each image came from. That name
    is only a LABEL for the evaluation later - the search does not use it.

    Returns a list of (image_path, folder_name) pairs.
    """
    image_list = []

    if not os.path.isdir(DATASET_DIR):
        print("ERROR: dataset folder not found:", DATASET_DIR)
        return image_list

    # sorted() so the order is always the same on every computer
    for folder_name in sorted(os.listdir(DATASET_DIR)):
        folder_path = os.path.join(DATASET_DIR, folder_name)
        if not os.path.isdir(folder_path):
            continue

        files_in_folder = []
        for root, _, files in os.walk(folder_path):
            for file_name in sorted(files):
                # accept only real image extensions
                if file_name.lower().endswith(VALID_EXTENSIONS):
                    files_in_folder.append(os.path.join(root, file_name))

        # For the demo we may use only N images of each folder
        # (see MAX_IMAGES_PER_CLASS in config.py). With the current dataset
        # that setting is None, so all 40 images of every folder are used.
        #
        # If a limit IS set, we do NOT simply take the first N files: we take
        # every "step"-th file, so the selection is spread over the whole
        # folder instead of being all the images whose name starts with "1".
        #
        # We choose 50% more candidates than we need, because a few of them
        # may be thrown away later for having no contrast (see build_index).
        if MAX_IMAGES_PER_CLASS is not None:
            wanted = int(MAX_IMAGES_PER_CLASS * 1.5)
            if len(files_in_folder) > wanted:
                step = len(files_in_folder) // wanted
                files_in_folder = files_in_folder[::step][:wanted]

        for file_path in files_in_folder:
            image_list.append((file_path, folder_name))

    return image_list


def build_index(progress_callback=None):
    """
    Calculate the features of every dataset image and save them to disk.

    progress_callback is an optional function(done, total) that we call so
    the GUI or the console can show the progress.

    If one image is broken we simply skip it and continue with the next one,
    so a single bad file can never stop the indexing.
    """
    image_list = list_dataset_images()
    total = len(image_list)

    if total == 0:
        print("ERROR: no images found in the dataset folder.")
        return []

    print("Indexing %d images ..." % total)
    database = []
    skipped = 0

    # How many usable images we already have from each folder.
    # This is only for limiting the size of the demo index - it has nothing
    # to do with the search itself.
    used_per_folder = {}

    for index, (image_path, folder_name) in enumerate(image_list):
        # stop taking images from a folder once we have enough usable ones
        if MAX_IMAGES_PER_CLASS is not None:
            if used_per_folder.get(folder_name, 0) >= MAX_IMAGES_PER_CLASS:
                continue

        gray, usable = load_and_preprocess(image_path)

        # the file is broken or not an image
        if gray is None:
            skipped += 1
            continue

        # the image is almost black / has no contrast -> it would only
        # produce noise, so we do not put it into the index
        if not usable:
            skipped += 1
            continue

        try:
            features = extract_all_features(gray)
        except Exception as error:
            # something unexpected happened with this image -> skip it
            print("  skipped (error):", image_path, "-", error)
            skipped += 1
            continue

        entry = {
            "path": image_path,
            "label": folder_name,        # only used for evaluation, NOT for searching
            "texture": features["texture"],
            "shape": features["shape"],
            "orb": features["orb"],
            "keypoints": features["keypoints"],
        }
        database.append(entry)
        used_per_folder[folder_name] = used_per_folder.get(folder_name, 0) + 1

        if progress_callback is not None:
            progress_callback(index + 1, total)

        # print progress in the console every 50 images
        if (index + 1) % 50 == 0:
            print("  %d / %d" % (index + 1, total))

    print("Indexing finished: %d images indexed, %d skipped "
          "(broken files or images without contrast)." % (len(database), skipped))
    save_index(database)
    return database


def current_settings():
    """
    The settings that decide what the saved features look like.

    If any of these changes, the features in features.pkl no longer match
    what the code would calculate now, so the index must be rebuilt. The
    format version alone is not enough: changing LBP_GRID or LBP_SCALES in
    config.py changes the LENGTH of the texture descriptor, and comparing a
    1024-number descriptor with a 2048-number one would crash.
    """
    return {
        "image_size": IMAGE_SIZE,
        "lbp_grid": LBP_GRID,
        "lbp_scales": list(LBP_SCALES),
        "use_roi": USE_ROI,
        "shape_descriptor": SHAPE_DESCRIPTOR,
        "local_descriptor": LOCAL_DESCRIPTOR,
    }


def save_index(database):
    """
    Save the feature database into features/features.pkl with pickle.

    We do not save the bare list. We save a small dictionary that also
    contains the format version and the settings the features were made
    with, so that load_index() can notice when the file is out of date.
    """
    os.makedirs(FEATURES_DIR, exist_ok=True)

    file_content = {
        "version": FEATURES_FORMAT_VERSION,
        "settings": current_settings(),
        "images": database,
    }

    with open(FEATURES_FILE, "wb") as file_handle:
        pickle.dump(file_content, file_handle)
    print("Features saved to:", FEATURES_FILE)


def load_index():
    """
    Load features/features.pkl if it exists.
    Returns None when there is no saved index yet or the file is damaged.
    """
    if not os.path.exists(FEATURES_FILE):
        return None

    try:
        with open(FEATURES_FILE, "rb") as file_handle:
            file_content = pickle.load(file_handle)
    except Exception as error:
        print("Could not read the feature file (%s). It will be rebuilt." % error)
        return None

    # An old features.pkl (written before the project was changed to
    # distances) is just a list, or carries an older version number. Its
    # features do not match the current code, so we rebuild it.
    if not isinstance(file_content, dict) or "images" not in file_content:
        print("The feature file has the old format. It will be rebuilt.")
        return None

    if file_content.get("version") != FEATURES_FORMAT_VERSION:
        print("The feature file was made by an older version of the code "
              "(found version %s, need %d). It will be rebuilt."
              % (file_content.get("version"), FEATURES_FORMAT_VERSION))
        return None

    # The descriptor settings must match too. Changing LBP_GRID or
    # LBP_SCALES in config.py changes the length of the texture descriptor,
    # and an old file would make the comparison crash.
    saved_settings = file_content.get("settings")
    if saved_settings != current_settings():
        print("The descriptor settings changed since the feature file was "
              "made (saved: %s, now: %s). It will be rebuilt."
              % (saved_settings, current_settings()))
        return None

    database = file_content["images"]
    print("Loaded %d indexed images from %s" % (len(database), FEATURES_FILE))
    return database


def get_index(force_rebuild=False):
    """
    Return the feature database: load it from disk if possible,
    otherwise calculate it once and save it.
    """
    if not force_rebuild:
        database = load_index()
        if database:
            return database

    return build_index()


# =====================================================================
# THE ACTUAL SEARCH
# =====================================================================

def search(query_gray, database, top_k=TOP_K, debug=False):
    """
    Search the TOP K most similar images for one query image.

    Everything here works with DISTANCES: a SMALL distance means SIMILAR.

    Steps:
        1. calculate the three descriptors of the query image
        2. calculate the automatic weights FROM THE QUERY IMAGE
        3. compare the query with EVERY image in the database and collect
           three separate distances (the label is never looked at here)
        4. normalise the three distance lists so they are comparable
        5. combine them with the weights into one final distance
        6. sort ASCENDING (smallest distance first) and return the first K

    NOTE: the query image is NOT removed from the results. If you pick an
    image that is also inside the dataset, it will appear as result #1 with a
    distance of 0. That is correct behaviour and a good way to show that the
    system works.

    Returns (results, weights).
    """
    # --- 1. features of the query
    query_features = extract_all_features(query_gray)

    # --- 2. automatic weights, calculated only from the query image
    weights, raw_importance = compute_weights(query_gray, return_raw=True)

    # --- 3. three separate distances to every database image.
    # We keep them in three plain lists, because step 4 needs to look at all
    # the values of one descriptor at the same time.
    texture_values = []
    shape_values = []
    local_values = []

    for entry in database:
        texture_values.append(texture_distance(query_features["texture"], entry["texture"]))
        shape_values.append(shape_distance(query_features["shape"], entry["shape"]))
        local_values.append(local_distance(query_features["orb"], entry["orb"]))

    # --- 4 + 5. normalise and combine into the final distance
    final_values, texture_norm, shape_norm, local_norm = combine_distances(
        texture_values, shape_values, local_values, weights)

    results = []
    for index, entry in enumerate(database):
        results.append({
            "path": entry["path"],
            "label": entry["label"],     # carried along ONLY for the evaluation
            # the raw distances (each already inside 0..1)
            "texture_distance": texture_values[index],
            "shape_distance": shape_values[index],
            "local_distance": local_values[index],
            # the normalised ones that were actually combined
            "texture_normalised": float(texture_norm[index]),
            "shape_normalised": float(shape_norm[index]),
            "local_normalised": float(local_norm[index]),
            "final_distance": float(final_values[index]),
        })

    # --- 6. sort ASCENDING: the smallest final distance is the best match
    results.sort(key=lambda item: item["final_distance"])

    top_results = results[:top_k]

    if debug:
        print_debug(query_features, weights, raw_importance, top_results)

    return top_results, weights


def print_debug(query_features, weights, raw_importance, top_results):
    """
    Print what happened during one search, so we can check that the ranking
    is really calculated the way we think it is.
    """
    print()
    print("=" * 78)
    print("QUERY")
    print("-" * 78)
    print("  ORB keypoints detected : %d" % query_features["keypoints"])
    print("  raw importance  texture: %.3f   shape: %.3f   local: %.3f"
          % (raw_importance["texture"], raw_importance["shape"], raw_importance["local"]))
    print("  normalised weights     : texture %.1f%%   shape %.1f%%   local %.1f%%"
          % (weights["texture"] * 100, weights["shape"] * 100, weights["local"] * 100))
    print("  (the three weights add up to %.3f)"
          % (weights["texture"] + weights["shape"] + weights["local"]))
    print()
    print("TOP %d RESULTS - sorted by final distance, SMALLEST first" % len(top_results))
    print("-" * 78)
    print("  %-4s %-9s %-9s %-9s %-9s  %s"
          % ("rank", "final", "texture", "shape", "local", "image"))
    for rank, item in enumerate(top_results, start=1):
        print("  %-4d %-9.4f %-9.4f %-9.4f %-9.4f  %s"
              % (rank,
                 item["final_distance"],
                 item["texture_distance"],
                 item["shape_distance"],
                 item["local_distance"],
                 item["path"]))
    print("=" * 78)
    print()


# =====================================================================
# OPTIONAL EVALUATION (runs AFTER the search, never influences it)
# =====================================================================

def precision_at_k(results, query_label):
    """
    Precision@K = how many of the K returned images come from the same
    folder (category) as the query image, divided by K.

    This is only a way to MEASURE how good the retrieval was. It is
    calculated after the results are already fixed, so it cannot change
    the ranking in any way.

    Returns None when we do not know the class of the query image
    (for example when the user picks a photo from outside the dataset).
    """
    if query_label is None or len(results) == 0:
        return None

    correct = 0
    for item in results:
        if item["label"] == query_label:
            correct += 1

    return correct / float(len(results))


def guess_query_label(query_path):
    """
    If the chosen query image lies inside the dataset, its parent folder
    tells us the category. We use this ONLY to be able to show Precision@10.

    Returns the folder name, or None if the image comes from somewhere else.
    """
    if not query_path:
        return None

    parent_folder = os.path.basename(os.path.dirname(os.path.abspath(query_path)))
    known_classes = [name for name in os.listdir(DATASET_DIR)
                     if os.path.isdir(os.path.join(DATASET_DIR, name))]

    if parent_folder in known_classes:
        return parent_folder
    return None

"""
database.py

Builds the feature index and performs image search.
"""

import os
import pickle

from config import (
    DATASET_DIR, FEATURES_DIR, FEATURES_FILE,
    VALID_EXTENSIONS, MAX_IMAGES_PER_CLASS, TOP_K,
    FEATURES_FORMAT_VERSION, IMAGE_SIZE, LBP_GRID, LBP_SCALES,
    USE_ROI, SHAPE_DESCRIPTOR, LOCAL_DESCRIPTOR
)

from preprocessing import load_and_preprocess
from feature_extractor import extract_all_features
from distances import (
    texture_distance,
    shape_distance,
    local_distance,
    combine_distances
)
from weighting import compute_weights


# ============================================================
# DATASET
# ============================================================

def list_dataset_images():
    """Return all dataset images as (path, label) pairs."""

    image_list = []

    if not os.path.isdir(DATASET_DIR):
        print("ERROR: dataset folder not found:", DATASET_DIR)
        return image_list

    for folder_name in sorted(os.listdir(DATASET_DIR)):
        folder_path = os.path.join(DATASET_DIR, folder_name)

        if not os.path.isdir(folder_path):
            continue

        files_in_folder = []

        for root, _, files in os.walk(folder_path):
            for file_name in sorted(files):
                if file_name.lower().endswith(VALID_EXTENSIONS):
                    files_in_folder.append(
                        os.path.join(root, file_name)
                    )

        # Optional limit for demo purposes
        if MAX_IMAGES_PER_CLASS is not None:
            wanted = int(MAX_IMAGES_PER_CLASS * 1.5)

            if len(files_in_folder) > wanted:
                step = len(files_in_folder) // wanted
                files_in_folder = files_in_folder[::step][:wanted]

        for file_path in files_in_folder:
            image_list.append((file_path, folder_name))

    return image_list


# ============================================================
# FEATURE INDEX
# ============================================================

def build_index(progress_callback=None):
    """Extract features from all dataset images and save the index."""

    image_list = list_dataset_images()
    total = len(image_list)

    if total == 0:
        print("ERROR: no images found in the dataset folder.")
        return []

    print("Indexing %d images ..." % total)

    database = []
    skipped = 0
    used_per_folder = {}

    for index, (image_path, folder_name) in enumerate(image_list):

        if MAX_IMAGES_PER_CLASS is not None:
            if used_per_folder.get(folder_name, 0) >= MAX_IMAGES_PER_CLASS:
                continue

        gray, usable = load_and_preprocess(image_path)

        if gray is None or not usable:
            skipped += 1
            continue

        try:
            features = extract_all_features(gray)
        except Exception as error:
            print("Skipped:", image_path, "-", error)
            skipped += 1
            continue

        entry = {
            "path": image_path,
            "label": folder_name,
            "texture": features["texture"],
            "shape": features["shape"],
            "orb": features["orb"],
            "keypoints": features["keypoints"],
        }

        database.append(entry)

        used_per_folder[folder_name] = (
            used_per_folder.get(folder_name, 0) + 1
        )

        if progress_callback is not None:
            progress_callback(index + 1, total)

        if (index + 1) % 50 == 0:
            print("%d / %d" % (index + 1, total))

    print(
        "Indexing finished: %d images indexed, %d skipped."
        % (len(database), skipped)
    )

    save_index(database)

    return database


def current_settings():
    """Settings that affect the saved feature vectors."""

    return {
        "image_size": IMAGE_SIZE,
        "lbp_grid": LBP_GRID,
        "lbp_scales": list(LBP_SCALES),
        "use_roi": USE_ROI,
        "shape_descriptor": SHAPE_DESCRIPTOR,
        "local_descriptor": LOCAL_DESCRIPTOR,
    }


def save_index(database):
    """Save the feature index to features.pkl."""

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
    """Load the saved feature index if it is still valid."""

    if not os.path.exists(FEATURES_FILE):
        return None

    try:
        with open(FEATURES_FILE, "rb") as file_handle:
            file_content = pickle.load(file_handle)

    except Exception as error:
        print("Could not read the feature file:", error)
        return None

    if not isinstance(file_content, dict) or "images" not in file_content:
        print("Old feature format detected. Rebuilding index.")
        return None

    if file_content.get("version") != FEATURES_FORMAT_VERSION:
        print("Feature version changed. Rebuilding index.")
        return None

    if file_content.get("settings") != current_settings():
        print("Descriptor settings changed. Rebuilding index.")
        return None

    database = file_content["images"]

    print(
        "Loaded %d indexed images from %s"
        % (len(database), FEATURES_FILE)
    )

    return database


def get_index(force_rebuild=False):
    """Load the index or build it if needed."""

    if not force_rebuild:
        database = load_index()

        if database:
            return database

    return build_index()


# ============================================================
# SEARCH
# ============================================================

def search(query_gray, database, top_k=TOP_K, debug=False):
    """
    Search for the most similar images.

    The three descriptors are compared separately.
    Their distances are normalized and combined using
    the automatic weights calculated from the query image.
    """

    query_features = extract_all_features(query_gray)

    weights, raw_importance = compute_weights(
        query_gray,
        return_raw=True
    )

    texture_values = []
    shape_values = []
    local_values = []

    for entry in database:
        texture_values.append(
            texture_distance(
                query_features["texture"],
                entry["texture"]
            )
        )

        shape_values.append(
            shape_distance(
                query_features["shape"],
                entry["shape"]
            )
        )

        local_values.append(
            local_distance(
                query_features["orb"],
                entry["orb"]
            )
        )

    final_values, texture_norm, shape_norm, local_norm = (
        combine_distances(
            texture_values,
            shape_values,
            local_values,
            weights
        )
    )

    results = []

    for index, entry in enumerate(database):
        results.append({
            "path": entry["path"],
            "label": entry["label"],

            "texture_distance": texture_values[index],
            "shape_distance": shape_values[index],
            "local_distance": local_values[index],

            "texture_normalised": float(texture_norm[index]),
            "shape_normalised": float(shape_norm[index]),
            "local_normalised": float(local_norm[index]),

            "final_distance": float(final_values[index]),
        })

    # Smaller final distance = more similar
    results.sort(
        key=lambda item: item["final_distance"]
    )

    top_results = results[:top_k]

    if debug:
        print_debug(
            query_features,
            weights,
            raw_importance,
            top_results
        )

    return top_results, weights


# ============================================================
# DEBUG OUTPUT
# ============================================================

def print_debug(query_features, weights, raw_importance, top_results):
    """Print useful information about one search."""

    print()
    print("=" * 70)

    print("QUERY")
    print("ORB keypoints:", query_features["keypoints"])

    print(
        "Raw importance: texture %.3f, shape %.3f, local %.3f"
        % (
            raw_importance["texture"],
            raw_importance["shape"],
            raw_importance["local"]
        )
    )

    print(
        "Weights: texture %.1f%%, shape %.1f%%, local %.1f%%"
        % (
            weights["texture"] * 100,
            weights["shape"] * 100,
            weights["local"] * 100
        )
    )

    print()
    print("TOP RESULTS")

    for rank, item in enumerate(top_results, start=1):
        print(
            "%d. final=%.4f  T=%.4f  S=%.4f  L=%.4f  %s"
            % (
                rank,
                item["final_distance"],
                item["texture_distance"],
                item["shape_distance"],
                item["local_distance"],
                item["path"]
            )
        )

    print("=" * 70)
    print()


# ============================================================
# EVALUATION HELPERS
# ============================================================

def precision_at_k(results, query_label):
    """
    Return the fraction of Top-K results that belong
    to the same category as the query.
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
    Return the query category if the image is inside
    one of the dataset folders.
    """

    if not query_path:
        return None

    parent_folder = os.path.basename(
        os.path.dirname(
            os.path.abspath(query_path)
        )
    )

    known_classes = [
        name
        for name in os.listdir(DATASET_DIR)
        if os.path.isdir(
            os.path.join(DATASET_DIR, name)
        )
    ]

    if parent_folder in known_classes:
        return parent_folder

    return None
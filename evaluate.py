"""
evaluate.py
-----------
OPTIONAL evaluation script (not needed for the demo itself).

    python evaluate.py

It uses every image of the dataset as a query, runs a normal search, and then
checks how many of the Top 10 results came from the same folder as the query.

VERY IMPORTANT:
The folder names are used ONLY HERE, AFTER the search is already finished.
The search itself never sees them. This script only measures how good the
retrieval was - it cannot change the retrieval.

ONE DIFFERENCE TO THE GUI:
All dataset images are inside the index, so the query would always find
ITSELF first with distance 0. In the GUI that is wanted (it proves the system
works). But for an honest measurement we skip that one result, otherwise
every query would get 1 correct hit for free. This only happens here, in the
evaluation - the search function itself never removes anything.

Two numbers are printed:

    Precision@10 - of the 10 returned images, how many are the right category
    Precision@1  - was the very first result the right category

The random baseline is 1/5 = 0.20, because the dataset has 5 folders.
"""

import os
import glob

import numpy as np

import database
from preprocessing import load_and_preprocess
from feature_extractor import extract_all_features
from distances import (texture_distance, shape_distance, local_distance,
                       normalise_by_mean)
from weighting import compute_weights
from config import DATASET_DIR, TOP_K, SHAPE_DESCRIPTOR


def precision_of_ranking(distance_values, labels, query_label, skip_index):
    """
    Sort by distance (ASCENDING - smallest first), drop the query itself,
    and count how many of the first TOP_K have the right label.
    """
    order = np.argsort(distance_values)
    order = [i for i in order if i != skip_index][:TOP_K]

    hits = sum(1 for i in order if labels[i] == query_label)
    first_is_correct = len(order) > 0 and labels[order[0]] == query_label

    return hits / float(TOP_K), first_is_correct


def main():
    index_database = database.get_index()
    if not index_database:
        print("No index available.")
        return

    labels = [entry["label"] for entry in index_database]
    paths = [entry["path"] for entry in index_database]

    class_names = sorted(name for name in os.listdir(DATASET_DIR)
                         if os.path.isdir(os.path.join(DATASET_DIR, name)))

    print("Index: %d images,  %d categories" % (len(index_database), len(class_names)))
    print("Using every indexed image as a query ...\n")

    # we collect the results of the single descriptors as well, so we can see
    # which descriptor works best on this dataset
    precision_combined = []
    precision_texture = []
    precision_shape = []
    precision_local = []
    correct_first = 0
    per_class = {}

    for query_index, query_path in enumerate(paths):
        query_label = labels[query_index]

        gray, usable = load_and_preprocess(query_path)
        if gray is None:
            continue

        query_features = extract_all_features(gray)
        weights = compute_weights(gray)

        texture_values = []
        shape_values = []
        local_values = []
        for entry in index_database:
            texture_values.append(texture_distance(query_features["texture"], entry["texture"]))
            shape_values.append(shape_distance(query_features["shape"], entry["shape"]))
            local_values.append(local_distance(query_features["orb"], entry["orb"]))

        # the same normalisation the real search uses
        texture_normalised = normalise_by_mean(texture_values)
        shape_normalised = normalise_by_mean(shape_values)
        local_normalised = normalise_by_mean(local_values)

        final_values = (weights["texture"] * texture_normalised
                        + weights["shape"] * shape_normalised
                        + weights["local"] * local_normalised)

        combined, first_ok = precision_of_ranking(final_values, labels,
                                                  query_label, query_index)
        precision_combined.append(combined)
        if first_ok:
            correct_first += 1

        precision_texture.append(precision_of_ranking(
            np.array(texture_values), labels, query_label, query_index)[0])
        precision_shape.append(precision_of_ranking(
            np.array(shape_values), labels, query_label, query_index)[0])
        precision_local.append(precision_of_ranking(
            np.array(local_values), labels, query_label, query_index)[0])

        per_class.setdefault(query_label, []).append(combined)

    for class_name in class_names:
        if class_name in per_class:
            print("  %-10s Precision@10 = %.2f" % (class_name,
                                                   np.mean(per_class[class_name])))

    query_count = len(precision_combined)

    # Safety check: this happens when features.pkl was built on another
    # computer, so the stored image paths do not exist here. Without this
    # check the script would crash with a division by zero.
    if query_count == 0:
        print()
        print("No query image could be loaded.")
        print("The saved index probably points to files that do not exist on")
        print("this computer. Run  python main.py --rebuild  and try again.")
        return

    print()
    print("=" * 55)
    print("  queries                     : %d" % query_count)
    print("  random baseline  P@10       : %.3f" % (1.0 / len(class_names)))
    print("-" * 55)
    print("  texture (LBP) only   P@10   : %.3f" % np.mean(precision_texture))
    # the label follows SHAPE_DESCRIPTOR, so it stays correct when the
    # shape descriptor is switched in config.py
    shape_name = "HOG" if SHAPE_DESCRIPTOR == "hog" else "Hu"
    print("  shape (%-3s) only     P@10   : %.3f" % (shape_name, np.mean(precision_shape)))
    print("  local (ORB) only     P@10   : %.3f" % np.mean(precision_local))
    print("-" * 55)
    print("  COMBINED, dynamic w. P@10   : %.3f" % np.mean(precision_combined))
    print("  COMBINED, dynamic w. P@1    : %.3f" % (correct_first / float(query_count)))
    print("=" * 55)
    print()
    print("  (lower distance = more similar; the ranking is sorted ascending)")


if __name__ == "__main__":
    main()

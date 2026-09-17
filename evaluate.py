"""
evaluate.py

Evaluates the CBIR system using all indexed images as queries.
The query image itself is removed from the results before evaluation.
"""

import os
import numpy as np

import database

from preprocessing import load_and_preprocess
from feature_extractor import extract_all_features

from distances import (
    texture_distance,
    shape_distance,
    local_distance,
    normalise_by_mean
)

from weighting import compute_weights

from config import (
    DATASET_DIR,
    TOP_K,
    SHAPE_DESCRIPTOR,
    LOCAL_DESCRIPTOR
)


def precision_of_ranking(
        distance_values,
        labels,
        query_label,
        skip_index
):
    """Calculate Precision@K and whether the first result is correct."""

    order = np.argsort(distance_values)

    # Remove the query image itself
    order = [
        index
        for index in order
        if index != skip_index
    ][:TOP_K]

    hits = sum(
        1
        for index in order
        if labels[index] == query_label
    )

    first_is_correct = (
        len(order) > 0
        and labels[order[0]] == query_label
    )

    precision = hits / float(TOP_K)

    return precision, first_is_correct


def main():

    index_database = database.get_index()

    if not index_database:
        print("No index available.")
        return

    labels = [
        entry["label"]
        for entry in index_database
    ]

    paths = [
        entry["path"]
        for entry in index_database
    ]

    class_names = sorted(
        name
        for name in os.listdir(DATASET_DIR)
        if os.path.isdir(
            os.path.join(DATASET_DIR, name)
        )
    )

    print(
        "Index: %d images, %d categories"
        % (
            len(index_database),
            len(class_names)
        )
    )

    print("Using every indexed image as a query...\n")

    precision_combined = []
    precision_texture = []
    precision_shape = []
    precision_local = []

    correct_first = 0
    per_class = {}

    for query_index, query_path in enumerate(paths):

        query_label = labels[query_index]

        gray, usable = load_and_preprocess(query_path)

        if gray is None or not usable:
            continue

        query_features = extract_all_features(gray)
        weights = compute_weights(gray)

        texture_values = []
        shape_values = []
        local_values = []

        for entry in index_database:

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

        # Normalize descriptor distances
        texture_normalised = normalise_by_mean(
            texture_values
        )

        shape_normalised = normalise_by_mean(
            shape_values
        )

        local_normalised = normalise_by_mean(
            local_values
        )

        # Final combined distance
        final_values = (
            weights["texture"] * texture_normalised
            + weights["shape"] * shape_normalised
            + weights["local"] * local_normalised
        )

        combined, first_ok = precision_of_ranking(
            final_values,
            labels,
            query_label,
            query_index
        )

        precision_combined.append(combined)

        if first_ok:
            correct_first += 1

        texture_precision, _ = precision_of_ranking(
            np.asarray(texture_values),
            labels,
            query_label,
            query_index
        )

        shape_precision, _ = precision_of_ranking(
            np.asarray(shape_values),
            labels,
            query_label,
            query_index
        )

        local_precision, _ = precision_of_ranking(
            np.asarray(local_values),
            labels,
            query_label,
            query_index
        )

        precision_texture.append(texture_precision)
        precision_shape.append(shape_precision)
        precision_local.append(local_precision)

        per_class.setdefault(
            query_label,
            []
        ).append(combined)

    # Results for every category
    for class_name in class_names:

        if class_name in per_class:
            print(
                "%-10s Precision@10 = %.2f"
                % (
                    class_name,
                    np.mean(per_class[class_name])
                )
            )

    query_count = len(precision_combined)

    if query_count == 0:
        print("\nNo query image could be loaded.")
        print("Try rebuilding the feature index.")
        return

    shape_name = (
        "HOG"
        if SHAPE_DESCRIPTOR == "hog"
        else "Hu"
    )

    local_name = (
        "SIFT"
        if LOCAL_DESCRIPTOR == "sift"
        else "ORB"
    )

    random_baseline = 1.0 / len(class_names)

    print()
    print("=" * 55)

    print(
        "queries                     : %d"
        % query_count
    )

    print(
        "random baseline P@10        : %.3f"
        % random_baseline
    )

    print("-" * 55)

    print(
        "texture (LBP) only P@10     : %.3f"
        % np.mean(precision_texture)
    )

    print(
        "shape (%s) only P@10        : %.3f"
        % (
            shape_name,
            np.mean(precision_shape)
        )
    )

    print(
        "local (%s) only P@10        : %.3f"
        % (
            local_name,
            np.mean(precision_local)
        )
    )

    print("-" * 55)

    print(
        "COMBINED dynamic P@10       : %.3f"
        % np.mean(precision_combined)
    )

    print(
        "COMBINED dynamic P@1        : %.3f"
        % (
            correct_first
            / float(query_count)
        )
    )

    print("=" * 55)

    print(
        "\nLower distance = more similar."
    )


if __name__ == "__main__":
    main()
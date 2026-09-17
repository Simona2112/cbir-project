"""
main.py

Starts the CBIR application.
"""

import sys

import database
from gui import start_gui


def main():

    # Rebuild the feature index if --rebuild is used
    force_rebuild = "--rebuild" in sys.argv

    print("=" * 50)
    print("Content-Based Image Retrieval")
    print("(distance based: a SMALL distance means a SIMILAR image)")
    print("=" * 50)

    index_database = database.get_index(
        force_rebuild=force_rebuild
    )

    if not index_database:
        print(
            "No images could be indexed. "
            "Check the dataset folder."
        )
        return

    print(
        "Ready. %d images in the index."
        % len(index_database)
    )

    start_gui(index_database)


if __name__ == "__main__":
    main()
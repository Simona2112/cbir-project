"""
main.py
-------
Start the application.

    python main.py                -> use the saved index (or build it if missing)
    python main.py --rebuild      -> calculate all features again

Content-Based Image Retrieval - Digital Image Processing seminar project.
"""

import sys

import database
from gui import start_gui


def main():
    # "--rebuild" on the command line forces a new indexing
    force_rebuild = "--rebuild" in sys.argv

    print("=" * 60)
    print("Content-Based Image Retrieval")
    print("(distance based: a SMALL distance means a SIMILAR image)")
    print("=" * 60)

    # Load the feature database, or build it the first time.
    index_database = database.get_index(force_rebuild=force_rebuild)

    if not index_database:
        print("No images could be indexed. Check the 'dataset' folder.")
        return

    print("Ready. %d images in the index." % len(index_database))
    print("Opening the window ...")
    print("Every search prints its full calculation here in the console.")

    start_gui(index_database)


if __name__ == "__main__":
    main()

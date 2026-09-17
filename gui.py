"""
gui.py

Tkinter interface for the CBIR demo application.
"""

import os
import tkinter as tk
from tkinter import filedialog, messagebox

from PIL import Image, ImageTk

import database
from preprocessing import load_and_preprocess, load_image
from weighting import compute_weights
from config import TOP_K


PREVIEW_SIZE = 220
THUMBNAIL_SIZE = 110
RESULTS_AREA_HEIGHT = 330


def make_photo(path, size):
    """Load and resize an image for Tkinter."""

    try:
        image = Image.open(path).convert("RGB")
        image.thumbnail((size, size))
        return ImageTk.PhotoImage(image)

    except Exception:
        return None


class CBIRWindow:

    def __init__(self, root, index_database):

        self.root = root
        self.database = index_database

        self.query_path = None
        self.query_gray = None

        # Keep image references so Tkinter does not remove them
        self.result_photos = []
        self.query_photo = None

        self.build_widgets()


    # ============================================================
    # WINDOW LAYOUT
    # ============================================================

    def build_widgets(self):

        self.root.title("Content-Based Image Retrieval")
        self.root.configure(bg="#f0f0f0")

        tk.Label(
            self.root,
            text="Content-Based Image Retrieval",
            font=("Arial", 18, "bold"),
            bg="#f0f0f0"
        ).pack(pady=(12, 2))

        tk.Label(
            self.root,
            text="Texture (LBP) + Shape (HOG) + Local features (ORB)",
            font=("Arial", 10),
            fg="#555555",
            bg="#f0f0f0"
        ).pack(pady=(0, 10))

        top_frame = tk.Frame(
            self.root,
            bg="#f0f0f0"
        )

        top_frame.pack(
            padx=15,
            pady=5,
            fill="x"
        )

        # Query image
        left_frame = tk.Frame(
            top_frame,
            bg="#f0f0f0"
        )

        left_frame.pack(side="left")

        tk.Label(
            left_frame,
            text="Query image",
            font=("Arial", 11, "bold"),
            bg="#f0f0f0"
        ).pack()

        self.query_canvas = tk.Label(
            left_frame,
            text="no image selected",
            width=30,
            height=11,
            relief="sunken",
            bg="white"
        )

        self.query_canvas.pack(pady=5)

        self.query_name_label = tk.Label(
            left_frame,
            text="",
            font=("Arial", 9),
            fg="#555555",
            bg="#f0f0f0"
        )

        self.query_name_label.pack()

        # Buttons and information
        right_frame = tk.Frame(
            top_frame,
            bg="#f0f0f0"
        )

        right_frame.pack(
            side="left",
            padx=25,
            anchor="n"
        )

        self.choose_button = tk.Button(
            right_frame,
            text="Choose Query Image",
            font=("Arial", 11),
            width=22,
            command=self.choose_image
        )

        self.choose_button.pack(
            pady=(20, 6)
        )

        self.search_button = tk.Button(
            right_frame,
            text="Search Similar Images",
            font=("Arial", 11, "bold"),
            width=22,
            state="disabled",
            command=self.run_search
        )

        self.search_button.pack(pady=6)

        tk.Label(
            right_frame,
            text="Automatic descriptor weights",
            font=("Arial", 11, "bold"),
            bg="#f0f0f0"
        ).pack(pady=(18, 2))

        self.weights_label = tk.Label(
            right_frame,
            text="Texture: --\nShape: --\nLocal features: --",
            font=("Courier", 11),
            justify="left",
            bg="#f0f0f0"
        )

        self.weights_label.pack(pady=6)

        self.status_label = tk.Label(
            right_frame,
            text="",
            font=("Arial", 10),
            fg="#006600",
            bg="#f0f0f0"
        )

        self.status_label.pack(pady=8)

        tk.Label(
            self.root,
            text="Top %d results" % TOP_K,
            font=("Arial", 12, "bold"),
            bg="#f0f0f0"
        ).pack(pady=(12, 0))

        tk.Label(
            self.root,
            text="Lower final distance means more similar",
            font=("Arial", 9),
            fg="#555555",
            bg="#f0f0f0"
        ).pack(pady=(0, 4))

        self.build_scrollable_results()


    # ============================================================
    # SCROLLABLE RESULTS
    # ============================================================

    def build_scrollable_results(self):

        outer_frame = tk.Frame(
            self.root,
            bg="#f0f0f0"
        )

        outer_frame.pack(
            padx=15,
            pady=(0, 12),
            fill="both",
            expand=True
        )

        self.results_canvas = tk.Canvas(
            outer_frame,
            bg="#f0f0f0",
            height=RESULTS_AREA_HEIGHT,
            highlightthickness=0
        )

        self.results_canvas.pack(
            side="left",
            fill="both",
            expand=True
        )

        scrollbar = tk.Scrollbar(
            outer_frame,
            orient="vertical",
            command=self.results_canvas.yview
        )

        scrollbar.pack(
            side="right",
            fill="y"
        )

        self.results_canvas.configure(
            yscrollcommand=scrollbar.set
        )

        self.results_frame = tk.Frame(
            self.results_canvas,
            bg="#f0f0f0"
        )

        self.results_canvas.create_window(
            (0, 0),
            window=self.results_frame,
            anchor="nw"
        )

        self.results_frame.bind(
            "<Configure>",
            self.on_results_resize
        )

        self.results_canvas.bind(
            "<Enter>",
            self.bind_mousewheel
        )

        self.results_canvas.bind(
            "<Leave>",
            self.unbind_mousewheel
        )


    def on_results_resize(self, event):

        self.results_canvas.configure(
            scrollregion=self.results_canvas.bbox("all")
        )


    def bind_mousewheel(self, event):

        self.results_canvas.bind_all(
            "<MouseWheel>",
            self.on_mousewheel
        )

        self.results_canvas.bind_all(
            "<Button-4>",
            self.on_mousewheel
        )

        self.results_canvas.bind_all(
            "<Button-5>",
            self.on_mousewheel
        )


    def unbind_mousewheel(self, event):

        self.results_canvas.unbind_all(
            "<MouseWheel>"
        )

        self.results_canvas.unbind_all(
            "<Button-4>"
        )

        self.results_canvas.unbind_all(
            "<Button-5>"
        )


    def on_mousewheel(self, event):

        if event.num == 4:
            step = -1

        elif event.num == 5:
            step = 1

        else:
            step = -1 if event.delta > 0 else 1

        self.results_canvas.yview_scroll(
            step,
            "units"
        )


    # ============================================================
    # QUERY IMAGE
    # ============================================================

    def choose_image(self):

        path = filedialog.askopenfilename(
            title="Choose a query image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp"),
                ("All files", "*.*")
            ]
        )

        if not path:
            return

        image = load_image(path)

        if image is None:
            messagebox.showerror(
                "Error",
                "This file could not be opened as an image."
            )
            return

        self.query_path = path

        self.query_gray, usable = load_and_preprocess(
            path
        )

        if not usable:
            messagebox.showwarning(
                "Low contrast",
                "This image has very low contrast.\n"
                "The results may not be reliable."
            )

        self.query_photo = make_photo(
            path,
            PREVIEW_SIZE
        )

        if self.query_photo is not None:
            self.query_canvas.configure(
                image=self.query_photo,
                text="",
                width=PREVIEW_SIZE,
                height=PREVIEW_SIZE
            )

        self.query_name_label.configure(
            text=os.path.basename(path)
        )

        self.show_weights(
            compute_weights(self.query_gray)
        )

        self.search_button.configure(
            state="normal"
        )

        self.status_label.configure(
            text="Ready to search.",
            fg="#006600"
        )

        self.clear_results()


    def show_weights(self, weights):

        text = (
            "Texture:        %5.1f %%\n"
            "Shape:          %5.1f %%\n"
            "Local features: %5.1f %%"
        ) % (
            weights["texture"] * 100,
            weights["shape"] * 100,
            weights["local"] * 100
        )

        self.weights_label.configure(
            text=text
        )


    # ============================================================
    # SEARCH
    # ============================================================

    def run_search(self):

        if self.query_gray is None:
            return

        self.status_label.configure(
            text="Searching...",
            fg="#aa5500"
        )

        self.search_button.configure(
            state="disabled"
        )

        self.root.update()

        try:
            results, weights = database.search(
                self.query_gray,
                self.database,
                debug=True
            )

        except Exception as error:

            messagebox.showerror(
                "Error",
                "The search failed:\n%s" % error
            )

            self.search_button.configure(
                state="normal"
            )

            self.status_label.configure(
                text=""
            )

            return

        self.show_weights(weights)
        self.show_results(results)

        query_label = database.guess_query_label(
            self.query_path
        )

        precision = database.precision_at_k(
            results,
            query_label
        )

        if precision is None:

            status = (
                "Done. Query is not from the dataset,\n"
                "so Precision@%d is not available."
                % TOP_K
            )

        else:

            status = (
                "Done. Query folder: %s\n"
                "Precision@%d = %.0f %%"
                % (
                    query_label,
                    TOP_K,
                    precision * 100
                )
            )

        self.status_label.configure(
            text=status,
            fg="#006600"
        )

        self.search_button.configure(
            state="normal"
        )


    # ============================================================
    # RESULTS
    # ============================================================

    def clear_results(self):

        for widget in self.results_frame.winfo_children():
            widget.destroy()

        self.result_photos = []

        self.results_canvas.yview_moveto(0)


    def show_results(self, results):

        self.clear_results()

        for position, item in enumerate(results):

            row = position // 5
            column = position % 5

            cell = tk.Frame(
                self.results_frame,
                bg="white",
                relief="ridge",
                borderwidth=1
            )

            cell.grid(
                row=row,
                column=column,
                padx=5,
                pady=5,
                sticky="n"
            )

            photo = make_photo(
                item["path"],
                THUMBNAIL_SIZE
            )

            if photo is None:

                tk.Label(
                    cell,
                    text="Image not readable",
                    width=14,
                    height=6,
                    bg="white"
                ).pack()

            else:

                self.result_photos.append(photo)

                tk.Label(
                    cell,
                    image=photo,
                    bg="white"
                ).pack()

            tk.Label(
                cell,
                text="#%d" % (position + 1),
                font=("Arial", 11, "bold"),
                bg="white"
            ).pack()

            tk.Label(
                cell,
                text=item["label"],
                font=("Arial", 9),
                fg="#444444",
                bg="white"
            ).pack()

            tk.Label(
                cell,
                text="Final distance: %.3f"
                     % item["final_distance"],
                font=("Arial", 9, "bold"),
                bg="white"
            ).pack(pady=(3, 0))

            tk.Label(
                cell,
                text=(
                    "T: %.3f\n"
                    "S: %.3f\n"
                    "L: %.3f"
                    % (
                        item["texture_distance"],
                        item["shape_distance"],
                        item["local_distance"]
                    )
                ),
                font=("Courier", 8),
                fg="#666666",
                bg="white",
                justify="left"
            ).pack(pady=(0, 4))


def start_gui(index_database):
    """Start the Tkinter application."""

    root = tk.Tk()

    root.geometry("900x820")
    root.minsize(820, 600)

    CBIRWindow(
        root,
        index_database
    )

    root.mainloop()
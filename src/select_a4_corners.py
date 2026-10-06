"""
Manual A4 Planar Boundary Selection

This script allows the four physical corners of the A4 planar object
to be selected manually in View 1.

The selected points are saved as:
    outputs/sfm/a4_boundary_corners.json

A verification image is saved as:
    outputs/sfm/a4_boundary_selected.jpg

Corner order:
    1. Top-left
    2. Top-right
    3. Bottom-right
    4. Bottom-left
"""

from pathlib import Path
import json

import cv2
import numpy as np


# =====================================================================
# PATHS
# =====================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

IMAGE_PATH = (
    PROJECT_ROOT
    / "data"
    / "sfm"
    / "view1.jpeg"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "sfm"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

CORNERS_JSON = (
    OUTPUT_DIR
    / "a4_boundary_corners.json"
)

OUTPUT_IMAGE = (
    OUTPUT_DIR
    / "a4_boundary_selected.jpg"
)


# =====================================================================
# GLOBAL STATE
# =====================================================================

selected_points = []

display_image = None
original_image = None


# =====================================================================
# MOUSE CALLBACK
# =====================================================================

def mouse_callback(
    event,
    x,
    y,
    flags,
    param
):
    """
    Record one point whenever the user clicks.

    The user must select exactly four points in this order:

        1. Top-left
        2. Top-right
        3. Bottom-right
        4. Bottom-left
    """

    global selected_points
    global display_image

    if event != cv2.EVENT_LBUTTONDOWN:
        return

    # Do not allow more than four points.
    if len(selected_points) >= 4:
        return

    selected_points.append(
        (x, y)
    )

    print(
        f"Point {len(selected_points)}: "
        f"x={x}, y={y}"
    )

    redraw()


# =====================================================================
# DRAWING
# =====================================================================

def redraw():
    """Redraw the image with selected points."""

    global display_image

    display_image = original_image.copy()

    labels = [
        "TL",
        "TR",
        "BR",
        "BL"
    ]

    # Draw selected points.
    for index, point in enumerate(
        selected_points
    ):

        x, y = point

        cv2.circle(
            display_image,
            (x, y),
            8,
            (0, 0, 255),
            -1
        )

        cv2.putText(
            display_image,
            labels[index],
            (x + 10, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
            cv2.LINE_AA
        )

    # Draw connecting lines once at least two points exist.
    if len(selected_points) >= 2:

        for i in range(
            len(selected_points) - 1
        ):

            cv2.line(
                display_image,
                selected_points[i],
                selected_points[i + 1],
                (0, 255, 0),
                4
            )

    # Close the quadrilateral after four points.
    if len(selected_points) == 4:

        cv2.line(
            display_image,
            selected_points[3],
            selected_points[0],
            (0, 255, 0),
            4
        )

    # Instructions.
    cv2.rectangle(
        display_image,
        (10, 10),
        (780, 85),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        display_image,
        "Click A4 corners: TL -> TR -> BR -> BL",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    cv2.putText(
        display_image,
        "Press R to reset | ENTER to save | ESC to cancel",
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )


# =====================================================================
# SAVE
# =====================================================================

def save_results():
    """Save selected corner coordinates and visualization."""

    if len(selected_points) != 4:

        raise RuntimeError(
            "Exactly four corners must be selected."
        )

    labels = [
        "top_left",
        "top_right",
        "bottom_right",
        "bottom_left"
    ]

    corners = {}

    for label, point in zip(
        labels,
        selected_points
    ):

        corners[label] = [
            int(point[0]),
            int(point[1])
        ]

    # ---------------------------------------------------------------
    # Save JSON
    # ---------------------------------------------------------------

    data = {
        "image": "view1.jpeg",
        "coordinate_system":
            "View 1 image pixel coordinates",
        "corner_order": labels,
        "corners": corners
    }

    with open(
        CORNERS_JSON,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            data,
            file,
            indent=4
        )

    # ---------------------------------------------------------------
    # Save image
    # ---------------------------------------------------------------

    cv2.imwrite(
        str(OUTPUT_IMAGE),
        display_image
    )

    print("\n" + "=" * 60)
    print("A4 CORNERS SAVED")
    print("=" * 60)

    for label, point in zip(
        labels,
        selected_points
    ):

        print(
            f"{label}: "
            f"x={point[0]}, "
            f"y={point[1]}"
        )

    print(
        f"\nJSON:"
    )

    print(
        CORNERS_JSON
    )

    print(
        "\nVerification image:"
    )

    print(
        OUTPUT_IMAGE
    )

    print("=" * 60)


# =====================================================================
# MAIN
# =====================================================================

def main():

    global original_image
    global display_image
    global selected_points

    if not IMAGE_PATH.exists():

        raise FileNotFoundError(
            f"Image not found:\n{IMAGE_PATH}"
        )

    original_image = cv2.imread(
        str(IMAGE_PATH)
    )

    if original_image is None:

        raise RuntimeError(
            "OpenCV could not read view1.jpeg."
        )

    height, width = original_image.shape[:2]

    print("=" * 60)
    print("CSC 8830 - A4 CORNER SELECTION")
    print("=" * 60)

    print(
        f"\nImage size: "
        f"{width} x {height}"
    )

    print(
        "\nSelect the four corners in this exact order:"
    )

    print(
        "1. Top-left"
    )

    print(
        "2. Top-right"
    )

    print(
        "3. Bottom-right"
    )

    print(
        "4. Bottom-left"
    )

    print(
        "\nBe precise: click directly on the four "
        "physical edges/corners of the paper."
    )

    print(
        "\nPress ENTER after selecting four points."
    )

    print(
        "Press R to reset."
    )

    print(
        "Press ESC to cancel."
    )

    redraw()

    window_name = (
        "CSC 8830 - Select A4 Corners"
    )

    cv2.namedWindow(
        window_name,
        cv2.WINDOW_NORMAL
    )

    cv2.resizeWindow(
        window_name,
        810,
        1080
    )

    cv2.setMouseCallback(
        window_name,
        mouse_callback
    )

    while True:

        cv2.imshow(
            window_name,
            display_image
        )

        key = cv2.waitKey(20) & 0xFF

        # -----------------------------------------------------------
        # ENTER
        # -----------------------------------------------------------

        if key in (
            13,
            10
        ):

            if len(selected_points) == 4:

                save_results()

                break

            print(
                f"\nYou selected "
                f"{len(selected_points)} points."
            )

            print(
                "Please select exactly four."
            )

        # -----------------------------------------------------------
        # RESET
        # -----------------------------------------------------------

        elif key in (
            ord("r"),
            ord("R")
        ):

            selected_points = []

            print(
                "\nPoints reset."
            )

            redraw()

        # -----------------------------------------------------------
        # ESC
        # -----------------------------------------------------------

        elif key == 27:

            print(
                "\nSelection cancelled."
            )

            break

    cv2.destroyAllWindows()


# =====================================================================
# ENTRY POINT
# =====================================================================

if __name__ == "__main__":
    main()
"""
Bilinear Interpolation Demonstration

Uses an actual consecutive frame pair from the selected videos,
computes dense Farneback optical flow, and demonstrates bilinear
interpolation of the flow vector at a subpixel location.

The numerical example is saved as JSON and CSV so it can be used
as evidence in the final report.
"""

from pathlib import Path
import csv
import json

import cv2
import numpy as np


# =========================================================
# Project paths
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

VIDEO_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "optical_flow"
    / "video_samples"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "tracking"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =========================================================
# Video and frame selection
# =========================================================

VIDEO_PATH = (
    VIDEO_DIR
    / "cars_traffic_30s.mp4"
)

# Same general frame pair used in the tracking validation.
FRAME_INDEX = 750


# =========================================================
# Farneback parameters
# =========================================================

PYRAMID_SCALE = 0.5
PYRAMID_LEVELS = 3
WINDOW_SIZE = 15
ITERATIONS = 3
POLYNOMIAL_N = 5
POLYNOMIAL_SIGMA = 1.2


# =========================================================
# Bilinear interpolation function
# =========================================================

def bilinear_interpolate(
    flow: np.ndarray,
    x: float,
    y: float,
):
    """
    Bilinearly interpolate a 2D optical-flow vector.

    The four surrounding flow vectors are:

        F00 -------- F10
         |            |
         |            |
        F01 -------- F11

    where:

        F00 = flow[y0, x0]
        F10 = flow[y0, x1]
        F01 = flow[y1, x0]
        F11 = flow[y1, x1]

    The interpolation formula is:

        F(x,y) =
            (1-a)(1-b)F00
            + a(1-b)F10
            + (1-a)bF01
            + abF11
    """

    height, width = flow.shape[:2]

    # Keep coordinates safely inside the image.
    x = float(
        np.clip(
            x,
            0,
            width - 1.001,
        )
    )

    y = float(
        np.clip(
            y,
            0,
            height - 1.001,
        )
    )

    x0 = int(np.floor(x))
    y0 = int(np.floor(y))

    x1 = min(
        x0 + 1,
        width - 1,
    )

    y1 = min(
        y0 + 1,
        height - 1,
    )

    # Fractional coordinates.
    a = x - x0
    b = y - y0

    # Four neighboring flow vectors.
    f00 = flow[y0, x0]
    f10 = flow[y0, x1]
    f01 = flow[y1, x0]
    f11 = flow[y1, x1]

    # Bilinear interpolation.
    interpolated = (
        (1 - a)
        * (1 - b)
        * f00
        +
        a
        * (1 - b)
        * f10
        +
        (1 - a)
        * b
        * f01
        +
        a
        * b
        * f11
    )

    return {
        "x": x,
        "y": y,
        "x0": x0,
        "x1": x1,
        "y0": y0,
        "y1": y1,
        "a": a,
        "b": b,
        "f00": f00,
        "f10": f10,
        "f01": f01,
        "f11": f11,
        "interpolated": interpolated,
    }


# =========================================================
# Read frame pair
# =========================================================

def read_frame_pair(
    video_path: Path,
    frame_index: int,
):
    capture = cv2.VideoCapture(
        str(video_path)
    )

    if not capture.isOpened():
        raise RuntimeError(
            f"Could not open: {video_path}"
        )

    capture.set(
        cv2.CAP_PROP_POS_FRAMES,
        frame_index,
    )

    success_1, frame_1 = capture.read()
    success_2, frame_2 = capture.read()

    capture.release()

    if not success_1 or not success_2:
        raise RuntimeError(
            f"Could not read frames "
            f"{frame_index} and "
            f"{frame_index + 1}"
        )

    return frame_1, frame_2


# =========================================================
# Main
# =========================================================

def main():

    print()
    print(
        "CSC 8830 - Module 6 "
        "Bilinear Interpolation Demo"
    )
    print("=" * 60)

    if not VIDEO_PATH.exists():
        raise FileNotFoundError(
            f"Video not found: {VIDEO_PATH}"
        )

    # -----------------------------------------------------
    # Load consecutive frames
    # -----------------------------------------------------

    frame_1, frame_2 = read_frame_pair(
        VIDEO_PATH,
        FRAME_INDEX,
    )

    gray_1 = cv2.cvtColor(
        frame_1,
        cv2.COLOR_BGR2GRAY,
    )

    gray_2 = cv2.cvtColor(
        frame_2,
        cv2.COLOR_BGR2GRAY,
    )

    height, width = gray_1.shape

    print(
        f"Video: {VIDEO_PATH.name}"
    )

    print(
        f"Frames: {FRAME_INDEX} and "
        f"{FRAME_INDEX + 1}"
    )

    print(
        f"Resolution: {width} x {height}"
    )

    # -----------------------------------------------------
    # Calculate dense optical flow
    # -----------------------------------------------------

    flow = cv2.calcOpticalFlowFarneback(
        gray_1,
        gray_2,
        None,
        PYRAMID_SCALE,
        PYRAMID_LEVELS,
        WINDOW_SIZE,
        ITERATIONS,
        POLYNOMIAL_N,
        POLYNOMIAL_SIGMA,
        0,
    )

    print(
        "Dense Farneback flow calculated."
    )

    # -----------------------------------------------------
    # Choose a real feature point
    # -----------------------------------------------------

    points = cv2.goodFeaturesToTrack(
        gray_1,
        maxCorners=20,
        qualityLevel=0.01,
        minDistance=20,
        blockSize=7,
        useHarrisDetector=False,
    )

    if points is None or len(points) == 0:
        raise RuntimeError(
            "No feature points detected."
        )

    # Use the first detected feature point.
    base_point = points[0, 0]

    base_x = float(base_point[0])
    base_y = float(base_point[1])

    # -----------------------------------------------------
    # Move to a subpixel position
    # -----------------------------------------------------

    # We intentionally use fractional coordinates so that
    # bilinear interpolation is genuinely required.
    x = base_x + 0.35
    y = base_y + 0.62

    # Keep the point safely inside the image.
    x = min(max(x, 1.0), width - 2.0)
    y = min(max(y, 1.0), height - 2.0)

    result = bilinear_interpolate(
        flow,
        x,
        y,
    )

    # -----------------------------------------------------
    # Extract values
    # -----------------------------------------------------

    x0 = result["x0"]
    x1 = result["x1"]
    y0 = result["y0"]
    y1 = result["y1"]

    a = result["a"]
    b = result["b"]

    f00 = result["f00"]
    f10 = result["f10"]
    f01 = result["f01"]
    f11 = result["f11"]

    interpolated = result["interpolated"]

    u = float(interpolated[0])
    v = float(interpolated[1])

    predicted_x = x + u
    predicted_y = y + v

    # -----------------------------------------------------
    # Display calculation
    # -----------------------------------------------------

    print()
    print("Selected subpixel location:")
    print(
        f"  x = {x:.4f}"
    )
    print(
        f"  y = {y:.4f}"
    )

    print()
    print("Neighboring flow vectors:")

    print(
        f"  F00 ({x0}, {y0}) = "
        f"({f00[0]:.6f}, {f00[1]:.6f})"
    )

    print(
        f"  F10 ({x1}, {y0}) = "
        f"({f10[0]:.6f}, {f10[1]:.6f})"
    )

    print(
        f"  F01 ({x0}, {y1}) = "
        f"({f01[0]:.6f}, {f01[1]:.6f})"
    )

    print(
        f"  F11 ({x1}, {y1}) = "
        f"({f11[0]:.6f}, {f11[1]:.6f})"
    )

    print()
    print(
        f"Fractional coordinates: "
        f"a = {a:.4f}, b = {b:.4f}"
    )

    print()
    print("Bilinear interpolation result:")

    print(
        f"  u = {u:.6f}"
    )

    print(
        f"  v = {v:.6f}"
    )

    print()
    print("Predicted next pixel location:")

    print(
        f"  x' = {predicted_x:.6f}"
    )

    print(
        f"  y' = {predicted_y:.6f}"
    )

    # -----------------------------------------------------
    # Save numerical evidence
    # -----------------------------------------------------

    output = {
        "video": VIDEO_PATH.name,
        "frame_t": FRAME_INDEX,
        "frame_t_plus_1": FRAME_INDEX + 1,
        "base_feature_point": {
            "x": base_x,
            "y": base_y,
        },
        "subpixel_location": {
            "x": x,
            "y": y,
        },
        "integer_neighbors": {
            "x0": x0,
            "x1": x1,
            "y0": y0,
            "y1": y1,
        },
        "fractional_coordinates": {
            "a": a,
            "b": b,
        },
        "flow_vectors": {
            "F00": [
                float(f00[0]),
                float(f00[1]),
            ],
            "F10": [
                float(f10[0]),
                float(f10[1]),
            ],
            "F01": [
                float(f01[0]),
                float(f01[1]),
            ],
            "F11": [
                float(f11[0]),
                float(f11[1]),
            ],
        },
        "interpolated_flow": {
            "u": u,
            "v": v,
        },
        "predicted_next_location": {
            "x": predicted_x,
            "y": predicted_y,
        },
    }

    json_path = (
        OUTPUT_DIR
        / "bilinear_interpolation_example.json"
    )

    with open(
        json_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            output,
            file,
            indent=4,
        )

    # -----------------------------------------------------
    # Save CSV
    # -----------------------------------------------------

    csv_path = (
        OUTPUT_DIR
        / "bilinear_interpolation_example.csv"
    )

    rows = [
        {
            "location": "F00",
            "x": x0,
            "y": y0,
            "u": float(f00[0]),
            "v": float(f00[1]),
        },
        {
            "location": "F10",
            "x": x1,
            "y": y0,
            "u": float(f10[0]),
            "v": float(f10[1]),
        },
        {
            "location": "F01",
            "x": x0,
            "y": y1,
            "u": float(f01[0]),
            "v": float(f01[1]),
        },
        {
            "location": "F11",
            "x": x1,
            "y": y1,
            "u": float(f11[0]),
            "v": float(f11[1]),
        },
    ]

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "location",
                "x",
                "y",
                "u",
                "v",
            ],
        )

        writer.writeheader()
        writer.writerows(rows)

    print()
    print("Saved:")
    print(
        f"  {json_path}"
    )
    print(
        f"  {csv_path}"
    )

    print()
    print("=" * 60)
    print(
        "BILINEAR INTERPOLATION DEMO COMPLETE"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()
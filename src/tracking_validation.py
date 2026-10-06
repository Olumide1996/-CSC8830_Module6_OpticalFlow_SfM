"""
Optical Flow Tracking Validation

Validates frame-to-frame motion tracking using:
1. Lucas-Kanade point tracking
2. Dense Farneback optical flow
3. Bilinear interpolation of the dense flow field

For one pair of consecutive frames from each video, the script:
- detects feature points,
- tracks them to the next frame,
- predicts their next positions using optical flow,
- compares predicted and actual positions,
- calculates pixel-level tracking error,
- classifies points as high-confidence inliers or outliers,
- saves annotated evidence images,
- saves CSV and JSON results.
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
# Video definitions
# =========================================================

VIDEOS = {
    "people_walking": VIDEO_DIR / "people_walking_30s.mp4",
    "cars_traffic": VIDEO_DIR / "cars_traffic_30s.mp4",
}


# =========================================================
# Frame-selection settings
# =========================================================

# We use a frame near the middle of each 30-second sample.
FRAME_FRACTION = 0.50


# =========================================================
# Feature-detection parameters
# =========================================================

MAX_CORNERS = 20
QUALITY_LEVEL = 0.01
MIN_DISTANCE = 20


# =========================================================
# Lucas-Kanade parameters
# =========================================================

LK_WINDOW = (21, 21)
LK_PYRAMID_LEVELS = 3

LK_CRITERIA = (
    cv2.TERM_CRITERIA_EPS
    | cv2.TERM_CRITERIA_COUNT,
    30,
    0.01,
)


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
# Tracking-quality threshold
# =========================================================

# A point is considered a high-confidence agreement when
# the Farneback optical-flow prediction is within 1 pixel
# of the Lucas-Kanade tracked position.
INLIER_THRESHOLD_PIXELS = 1.0


# =========================================================
# Helper functions
# =========================================================

def read_frame_pair(
    video_path: Path,
    frame_index: int,
):
    """
    Read frame t and frame t+1 from a video.
    """

    capture = cv2.VideoCapture(
        str(video_path)
    )

    if not capture.isOpened():
        raise RuntimeError(
            f"Could not open video: {video_path}"
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
            f"Could not read frame pair "
            f"{frame_index}, {frame_index + 1} "
            f"from {video_path}"
        )

    return frame_1, frame_2


def calculate_dense_flow(
    gray_1: np.ndarray,
    gray_2: np.ndarray,
) -> np.ndarray:
    """
    Calculate dense Farneback optical flow.
    """

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

    return flow


def bilinear_interpolate_flow(
    flow: np.ndarray,
    x: float,
    y: float,
) -> np.ndarray:
    """
    Bilinearly interpolate the dense optical-flow vector
    at a possibly subpixel location (x, y).

    The four neighboring vectors are:

        F00 -------- F10
         |            |
         |            |
        F01 -------- F11

    The interpolation weights are based on the fractional
    horizontal and vertical coordinates.
    """

    height, width = flow.shape[:2]

    # Keep the point safely inside the image.
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

    dx = x - x0
    dy = y - y0

    f00 = flow[y0, x0]
    f10 = flow[y0, x1]
    f01 = flow[y1, x0]
    f11 = flow[y1, x1]

    interpolated = (
        (1 - dx)
        * (1 - dy)
        * f00
        +
        dx
        * (1 - dy)
        * f10
        +
        (1 - dx)
        * dy
        * f01
        +
        dx
        * dy
        * f11
    )

    return interpolated


def draw_tracking_visualization(
    frame_1: np.ndarray,
    frame_2: np.ndarray,
    points_1: np.ndarray,
    actual_points: np.ndarray,
    predicted_points: np.ndarray,
    valid_indices: list,
    point_qualities: dict,
) -> tuple:
    """
    Create side-by-side tracking evidence images.

    Left:
        Starting frame with original points.

    Right:
        Next frame with:
        - actual tracked points
        - predicted optical-flow positions
        - connecting lines

    Green = high-confidence inlier
    Red = outlier
    """

    left = frame_1.copy()
    right = frame_2.copy()

    for idx in valid_indices:

        p1 = points_1[idx]
        actual = actual_points[idx]
        predicted = predicted_points[idx]

        quality = point_qualities.get(
            idx,
            "OUTLIER",
        )

        p1_int = (
            int(round(p1[0])),
            int(round(p1[1])),
        )

        actual_int = (
            int(round(actual[0])),
            int(round(actual[1])),
        )

        predicted_int = (
            int(round(predicted[0])),
            int(round(predicted[1])),
        )

        # Starting point
        cv2.circle(
            left,
            p1_int,
            6,
            (0, 255, 255),
            -1,
        )

        if quality == "INLIER":

            point_color = (0, 255, 0)

        else:

            point_color = (0, 0, 255)

        # Actual tracked point
        cv2.circle(
            right,
            actual_int,
            6,
            point_color,
            -1,
        )

        # Predicted point from optical flow
        cv2.circle(
            right,
            predicted_int,
            6,
            (255, 0, 0),
            2,
        )

        # Actual displacement
        cv2.line(
            right,
            p1_int,
            actual_int,
            point_color,
            1,
        )

        # Prediction displacement
        cv2.line(
            right,
            p1_int,
            predicted_int,
            (255, 0, 0),
            1,
        )

    cv2.putText(
        left,
        "Frame t: detected points",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        right,
        "Frame t+1: green=inlier, red=outlier, blue=prediction",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    combined = np.hstack(
        (left, right)
    )

    return left, right, combined


def save_csv(
    output_path: Path,
    rows: list,
):
    """
    Save point-level tracking results as CSV.
    """

    if not rows:
        return

    fieldnames = list(
        rows[0].keys()
    )

    with open(
        output_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)


# =========================================================
# Main processing
# =========================================================

def process_video(
    name: str,
    video_path: Path,
):
    print()
    print("=" * 60)
    print(
        f"Tracking validation: {name}"
    )
    print("=" * 60)

    if not video_path.exists():
        raise FileNotFoundError(
            f"Video not found: {video_path}"
        )

    # -----------------------------------------------------
    # Read video metadata
    # -----------------------------------------------------

    capture = cv2.VideoCapture(
        str(video_path)
    )

    if not capture.isOpened():
        raise RuntimeError(
            f"Could not open: {video_path}"
        )

    fps = capture.get(
        cv2.CAP_PROP_FPS
    )

    frame_count = int(
        capture.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    width = int(
        capture.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    height = int(
        capture.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    capture.release()

    # -----------------------------------------------------
    # Select consecutive frames
    # -----------------------------------------------------

    frame_index = int(
        frame_count * FRAME_FRACTION
    )

    frame_index = min(
        frame_index,
        frame_count - 2,
    )

    print(
        f"Selected frames: "
        f"{frame_index} and "
        f"{frame_index + 1}"
    )

    print(
        f"Resolution: {width} x {height}"
    )

    print(
        f"FPS: {fps:.2f}"
    )

    # -----------------------------------------------------
    # Load frame pair
    # -----------------------------------------------------

    frame_1, frame_2 = read_frame_pair(
        video_path,
        frame_index,
    )

    gray_1 = cv2.cvtColor(
        frame_1,
        cv2.COLOR_BGR2GRAY,
    )

    gray_2 = cv2.cvtColor(
        frame_2,
        cv2.COLOR_BGR2GRAY,
    )

    # -----------------------------------------------------
    # Detect feature points
    # -----------------------------------------------------

    points = cv2.goodFeaturesToTrack(
        gray_1,
        maxCorners=MAX_CORNERS,
        qualityLevel=QUALITY_LEVEL,
        minDistance=MIN_DISTANCE,
        blockSize=7,
        useHarrisDetector=False,
    )

    if points is None or len(points) == 0:
        raise RuntimeError(
            f"No trackable feature points found "
            f"for {name}."
        )

    points_1 = points.reshape(-1, 2)

    print(
        f"Detected feature points: "
        f"{len(points_1)}"
    )

    # -----------------------------------------------------
    # Lucas-Kanade tracking
    # -----------------------------------------------------

    points_2, status, errors = (
        cv2.calcOpticalFlowPyrLK(
            gray_1,
            gray_2,
            points,
            None,
            winSize=LK_WINDOW,
            maxLevel=LK_PYRAMID_LEVELS,
            criteria=LK_CRITERIA,
        )
    )

    if points_2 is None:
        raise RuntimeError(
            f"Lucas-Kanade tracking failed "
            f"for {name}."
        )

    points_2 = points_2.reshape(-1, 2)

    status = status.reshape(-1)

    # -----------------------------------------------------
    # Dense Farneback flow
    # -----------------------------------------------------

    print(
        "Computing dense Farneback flow "
        "for the selected frame pair..."
    )

    flow = calculate_dense_flow(
        gray_1,
        gray_2,
    )

    # -----------------------------------------------------
    # Compare actual vs predicted locations
    # -----------------------------------------------------

    rows = []

    valid_indices = []

    predicted_points = np.zeros_like(
        points_2,
        dtype=np.float32,
    )

    errors_in_pixels = []

    point_qualities = {}

    for i, point in enumerate(points_1):

        if status[i] != 1:
            continue

        x = float(point[0])
        y = float(point[1])

        # Check that the point is safely inside the
        # image for bilinear interpolation.
        if (
            x < 1
            or x >= width - 2
            or y < 1
            or y >= height - 2
        ):
            continue

        flow_vector = (
            bilinear_interpolate_flow(
                flow,
                x,
                y,
            )
        )

        u = float(
            flow_vector[0]
        )

        v = float(
            flow_vector[1]
        )

        predicted_x = x + u
        predicted_y = y + v

        actual_x = float(
            points_2[i, 0]
        )

        actual_y = float(
            points_2[i, 1]
        )

        error = float(
            np.sqrt(
                (
                    predicted_x
                    - actual_x
                ) ** 2
                +
                (
                    predicted_y
                    - actual_y
                ) ** 2
            )
        )

        predicted_points[i] = [
            predicted_x,
            predicted_y,
        ]

        errors_in_pixels.append(error)

        valid_indices.append(i)

        # Classify the point.
        if error <= INLIER_THRESHOLD_PIXELS:

            quality = "INLIER"

        else:

            quality = "OUTLIER"

        point_qualities[i] = quality

        rows.append(
            {
                "point_id": len(rows) + 1,
                "frame_t": frame_index,
                "frame_t_plus_1": (
                    frame_index + 1
                ),
                "x_t": round(x, 4),
                "y_t": round(y, 4),
                "flow_u": round(u, 4),
                "flow_v": round(v, 4),
                "predicted_x": round(
                    predicted_x,
                    4,
                ),
                "predicted_y": round(
                    predicted_y,
                    4,
                ),
                "actual_x": round(
                    actual_x,
                    4,
                ),
                "actual_y": round(
                    actual_y,
                    4,
                ),
                "tracking_error_pixels": round(
                    error,
                    4,
                ),
                "tracking_quality": quality,
            }
        )

    if not rows:
        raise RuntimeError(
            f"No valid tracked points remained "
            f"for {name}."
        )

    # -----------------------------------------------------
    # Calculate summary metrics
    # -----------------------------------------------------

    mean_error = float(
        np.mean(errors_in_pixels)
    )

    median_error = float(
        np.median(errors_in_pixels)
    )

    maximum_error = float(
        np.max(errors_in_pixels)
    )

    # Separate high-confidence points from outliers.
    inlier_errors = [
        error
        for error in errors_in_pixels
        if error <= INLIER_THRESHOLD_PIXELS
    ]

    outlier_errors = [
        error
        for error in errors_in_pixels
        if error > INLIER_THRESHOLD_PIXELS
    ]

    inlier_count = len(
        inlier_errors
    )

    outlier_count = len(
        outlier_errors
    )

    successful_points = len(rows)

    inlier_percentage = (
        (
            inlier_count
            / successful_points
        )
        * 100
        if successful_points > 0
        else 0.0
    )

    robust_mean_error = (
        float(
            np.mean(inlier_errors)
        )
        if inlier_errors
        else 0.0
    )

    robust_median_error = (
        float(
            np.median(inlier_errors)
        )
        if inlier_errors
        else 0.0
    )

    # -----------------------------------------------------
    # Output paths
    # -----------------------------------------------------

    csv_path = (
        OUTPUT_DIR
        / f"{name}_tracking_results.csv"
    )

    json_path = (
        OUTPUT_DIR
        / f"{name}_tracking_results.json"
    )

    frame_t_path = (
        OUTPUT_DIR
        / f"{name}_frame_t.png"
    )

    frame_t_plus_1_path = (
        OUTPUT_DIR
        / f"{name}_frame_t_plus_1_tracking.png"
    )

    comparison_path = (
        OUTPUT_DIR
        / f"{name}_tracking_comparison.png"
    )

    # -----------------------------------------------------
    # Create evidence visualization
    # -----------------------------------------------------

    (
        frame_t_visual,
        frame_t_plus_1_visual,
        combined_visual,
    ) = draw_tracking_visualization(
        frame_1,
        frame_2,
        points_1,
        points_2,
        predicted_points,
        valid_indices,
        point_qualities,
    )

    cv2.imwrite(
        str(frame_t_path),
        frame_t_visual,
    )

    cv2.imwrite(
        str(frame_t_plus_1_path),
        frame_t_plus_1_visual,
    )

    cv2.imwrite(
        str(comparison_path),
        combined_visual,
    )

    # -----------------------------------------------------
    # Save tabular results
    # -----------------------------------------------------

    save_csv(
        csv_path,
        rows,
    )

    # -----------------------------------------------------
    # Save JSON summary
    # -----------------------------------------------------

    summary = {
        "video_name": name,
        "video_file": str(video_path),
        "fps": float(fps),
        "frame_count": frame_count,
        "resolution": [
            width,
            height,
        ],
        "selected_frame_t": frame_index,
        "selected_frame_t_plus_1": (
            frame_index + 1
        ),
        "time_of_frame_t_seconds": (
            frame_index / fps
        ),
        "time_of_frame_t_plus_1_seconds": (
            (frame_index + 1) / fps
        ),
        "feature_detector": (
            "Shi-Tomasi goodFeaturesToTrack"
        ),
        "tracker": (
            "Lucas-Kanade pyramidal "
            "optical flow"
        ),
        "dense_flow": (
            "Farneback optical flow"
        ),
        "bilinear_interpolation": True,
        "detected_points": int(
            len(points_1)
        ),
        "successful_tracked_points": int(
            successful_points
        ),
        "inlier_threshold_pixels": (
            INLIER_THRESHOLD_PIXELS
        ),
        "inlier_points": int(
            inlier_count
        ),
        "outlier_points": int(
            outlier_count
        ),
        "inlier_percentage": (
            inlier_percentage
        ),
        "mean_tracking_error_pixels": (
            mean_error
        ),
        "median_tracking_error_pixels": (
            median_error
        ),
        "maximum_tracking_error_pixels": (
            maximum_error
        ),
        "robust_mean_tracking_error_pixels": (
            robust_mean_error
        ),
        "robust_median_tracking_error_pixels": (
            robust_median_error
        ),
        "csv_results": str(csv_path),
        "comparison_image": str(
            comparison_path
        ),
    }

    with open(
        json_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary,
            file,
            indent=4,
        )

    # -----------------------------------------------------
    # Print results
    # -----------------------------------------------------

    print()
    print(
        f"Detected points: "
        f"{len(points_1)}"
    )

    print(
        f"Successfully tracked: "
        f"{successful_points}"
    )

    print()
    print(
        f"Raw mean tracking error: "
        f"{mean_error:.4f} pixels"
    )

    print(
        f"Raw median tracking error: "
        f"{median_error:.4f} pixels"
    )

    print(
        f"Raw maximum tracking error: "
        f"{maximum_error:.4f} pixels"
    )

    print()
    print(
        f"High-confidence threshold: "
        f"{INLIER_THRESHOLD_PIXELS:.1f} pixel"
    )

    print(
        f"Inlier points: "
        f"{inlier_count}"
    )

    print(
        f"Outlier points: "
        f"{outlier_count}"
    )

    print(
        f"Inlier percentage: "
        f"{inlier_percentage:.1f}%"
    )

    print(
        f"Robust mean tracking error: "
        f"{robust_mean_error:.4f} pixels"
    )

    print(
        f"Robust median tracking error: "
        f"{robust_median_error:.4f} pixels"
    )

    print()
    print("Saved:")

    print(
        f"  {csv_path}"
    )

    print(
        f"  {json_path}"
    )

    print(
        f"  {frame_t_path}"
    )

    print(
        f"  {frame_t_plus_1_path}"
    )

    print(
        f"  {comparison_path}"
    )

    return summary


# =========================================================
# Program entry point
# =========================================================

def main():

    print()
    print(
        "CSC 8830 - Module 6 "
        "Tracking Validation"
    )

    print(
        "=" * 60
    )

    summaries = []

    for name, video_path in VIDEOS.items():

        summary = process_video(
            name,
            video_path,
        )

        summaries.append(summary)

    combined_path = (
        OUTPUT_DIR
        / "tracking_validation_summary.json"
    )

    with open(
        combined_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summaries,
            file,
            indent=4,
        )

    print()
    print(
        "=" * 60
    )

    print(
        "TRACKING VALIDATION COMPLETE"
    )

    print(
        "=" * 60
    )

    print(
        f"Summary saved to: "
        f"{combined_path}"
    )


if __name__ == "__main__":
    main()
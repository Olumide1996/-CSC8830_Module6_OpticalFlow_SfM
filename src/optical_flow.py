"""
Optical Flow Processing

Computes dense optical flow for the selected 30-second video samples
using the Farneback method and creates visualizations suitable for
the assignment demonstration and report.
"""

from pathlib import Path
import json

import cv2
import numpy as np


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_DIR = PROJECT_ROOT / "outputs" / "optical_flow" / "video_samples"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "optical_flow"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# Video definitions
# ---------------------------------------------------------

VIDEOS = {
    "people_walking": INPUT_DIR / "people_walking_30s.mp4",
    "cars_traffic": INPUT_DIR / "cars_traffic_30s.mp4",
}


# ---------------------------------------------------------
# Optical-flow parameters
# ---------------------------------------------------------

PYRAMID_SCALE = 0.5
PYRAMID_LEVELS = 3
WINDOW_SIZE = 15
ITERATIONS = 3
POLYNOMIAL_N = 5
POLYNOMIAL_SIGMA = 1.2

# For visualization only
ARROW_STEP = 32
ARROW_SCALE = 2.0

# Processing resolution.
# Smaller resolution makes dense optical flow significantly faster
# while preserving the motion pattern needed for visualization.
PROCESS_WIDTH = 640
PROCESS_HEIGHT = 360


# ---------------------------------------------------------
# Helper functions
# ---------------------------------------------------------

def resize_frame(frame: np.ndarray) -> np.ndarray:
    """Resize a frame to the processing resolution."""
    return cv2.resize(
        frame,
        (PROCESS_WIDTH, PROCESS_HEIGHT),
        interpolation=cv2.INTER_AREA,
    )


def calculate_flow(
    previous_gray: np.ndarray,
    current_gray: np.ndarray
) -> np.ndarray:
    """
    Calculate dense optical flow using the Farneback algorithm.

    Returns:
        flow: H x W x 2 array
              flow[..., 0] = horizontal motion (u)
              flow[..., 1] = vertical motion (v)
    """

    flow = cv2.calcOpticalFlowFarneback(
        previous_gray,
        current_gray,
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


def create_hsv_visualization(flow: np.ndarray) -> np.ndarray:
    """
    Convert optical-flow vectors into a color visualization.

    Hue represents direction.
    Saturation represents normalized motion strength.
    Value represents motion magnitude.
    """

    magnitude, angle = cv2.cartToPolar(
        flow[..., 0],
        flow[..., 1],
        angleInDegrees=False,
    )

    hsv = np.zeros(
        (flow.shape[0], flow.shape[1], 3),
        dtype=np.uint8,
    )

    # Direction -> Hue
    hsv[..., 0] = (
        angle * 180 / np.pi / 2
    ).astype(np.uint8)

    # Motion magnitude -> Saturation
    normalized_magnitude = cv2.normalize(
        magnitude,
        None,
        0,
        255,
        cv2.NORM_MINMAX,
    )

    hsv[..., 1] = normalized_magnitude.astype(np.uint8)

    # Keep visualization bright enough to see
    hsv[..., 2] = 255

    visualization = cv2.cvtColor(
        hsv,
        cv2.COLOR_HSV2BGR,
    )

    return visualization


def create_vector_visualization(
    frame: np.ndarray,
    flow: np.ndarray
) -> np.ndarray:
    """
    Overlay optical-flow motion vectors on a video frame.
    """

    visualization = frame.copy()

    height, width = flow.shape[:2]

    for y in range(0, height, ARROW_STEP):
        for x in range(0, width, ARROW_STEP):

            u = float(flow[y, x, 0])
            v = float(flow[y, x, 1])

            start_x = x
            start_y = y

            end_x = int(round(x + u * ARROW_SCALE))
            end_y = int(round(y + v * ARROW_SCALE))

            cv2.arrowedLine(
                visualization,
                (start_x, start_y),
                (end_x, end_y),
                (255, 255, 255),
                1,
                tipLength=0.25,
            )

    return visualization


def calculate_flow_statistics(flow: np.ndarray) -> dict:
    """
    Calculate simple motion statistics for the current frame pair.
    """

    magnitude, angle = cv2.cartToPolar(
        flow[..., 0],
        flow[..., 1],
        angleInDegrees=False,
    )

    return {
        "mean_magnitude": float(np.mean(magnitude)),
        "median_magnitude": float(np.median(magnitude)),
        "max_magnitude": float(np.max(magnitude)),
        "mean_horizontal_motion": float(
            np.mean(np.abs(flow[..., 0]))
        ),
        "mean_vertical_motion": float(
            np.mean(np.abs(flow[..., 1]))
        ),
    }


# ---------------------------------------------------------
# Main processing
# ---------------------------------------------------------

def process_video(name: str, input_path: Path) -> dict:

    print()
    print("=" * 60)
    print(f"Processing optical flow: {name}")
    print("=" * 60)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input video not found: {input_path}"
        )

    capture = cv2.VideoCapture(str(input_path))

    if not capture.isOpened():
        raise RuntimeError(
            f"Could not open video: {input_path}"
        )

    fps = capture.get(cv2.CAP_PROP_FPS)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    original_width = int(
        capture.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    original_height = int(
        capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    duration = frame_count / fps if fps > 0 else 0

    print(f"Input FPS:       {fps:.2f}")
    print(f"Input frames:    {frame_count}")
    print(
        f"Input resolution: {original_width} x {original_height}"
    )
    print(f"Input duration:  {duration:.2f} s")

    # -----------------------------------------------------
    # Create output paths
    # -----------------------------------------------------

    hsv_output_path = (
        OUTPUT_DIR / f"{name}_optical_flow_hsv.mp4"
    )

    vector_output_path = (
        OUTPUT_DIR / f"{name}_optical_flow_vectors.mp4"
    )

    sample_hsv_path = (
        OUTPUT_DIR / f"{name}_flow_sample.jpg"
    )

    sample_vector_path = (
        OUTPUT_DIR / f"{name}_vector_sample.jpg"
    )

    # Use original FPS but processing resolution
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    hsv_writer = cv2.VideoWriter(
        str(hsv_output_path),
        fourcc,
        fps,
        (PROCESS_WIDTH, PROCESS_HEIGHT),
    )

    vector_writer = cv2.VideoWriter(
        str(vector_output_path),
        fourcc,
        fps,
        (PROCESS_WIDTH, PROCESS_HEIGHT),
    )

    if not hsv_writer.isOpened():
        capture.release()
        raise RuntimeError(
            f"Could not create output video: {hsv_output_path}"
        )

    if not vector_writer.isOpened():
        capture.release()
        hsv_writer.release()
        raise RuntimeError(
            f"Could not create output video: {vector_output_path}"
        )

    # -----------------------------------------------------
    # Read first frame
    # -----------------------------------------------------

    success, first_frame = capture.read()

    if not success:
        capture.release()
        hsv_writer.release()
        vector_writer.release()

        raise RuntimeError(
            f"Could not read first frame from {input_path}"
        )

    first_frame = resize_frame(first_frame)

    previous_gray = cv2.cvtColor(
        first_frame,
        cv2.COLOR_BGR2GRAY,
    )

    # -----------------------------------------------------
    # Statistics
    # -----------------------------------------------------

    frame_index = 1
    statistics = []

    saved_sample = False

    # -----------------------------------------------------
    # Process frame pairs
    # -----------------------------------------------------

    while True:

        success, current_frame = capture.read()

        if not success:
            break

        current_frame = resize_frame(current_frame)

        current_gray = cv2.cvtColor(
            current_frame,
            cv2.COLOR_BGR2GRAY,
        )

        # ---------------------------------------------
        # Calculate optical flow
        # ---------------------------------------------

        flow = calculate_flow(
            previous_gray,
            current_gray,
        )

        # ---------------------------------------------
        # Create visualizations
        # ---------------------------------------------

        hsv_visualization = create_hsv_visualization(
            flow
        )

        vector_visualization = create_vector_visualization(
            current_frame,
            flow
        )

        # ---------------------------------------------
        # Save video frames
        # ---------------------------------------------

        hsv_writer.write(hsv_visualization)
        vector_writer.write(vector_visualization)

        # ---------------------------------------------
        # Statistics
        # ---------------------------------------------

        frame_stats = calculate_flow_statistics(flow)

        frame_stats["frame"] = frame_index

        statistics.append(frame_stats)

        # ---------------------------------------------
        # Save a representative frame
        # ---------------------------------------------

        if not saved_sample:

            sample_hsv_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            cv2.imwrite(
                str(sample_hsv_path),
                hsv_visualization,
            )

            cv2.imwrite(
                str(sample_vector_path),
                vector_visualization,
            )

            saved_sample = True

        # ---------------------------------------------
        # Progress reporting
        # ---------------------------------------------

        if frame_index % 100 == 0:

            progress = (
                frame_index / max(frame_count - 1, 1)
            ) * 100

            print(
                f"Processed frame {frame_index}/"
                f"{frame_count - 1} "
                f"({progress:.1f}%)"
            )

        previous_gray = current_gray
        frame_index += 1

    # -----------------------------------------------------
    # Release resources
    # -----------------------------------------------------

    capture.release()
    hsv_writer.release()
    vector_writer.release()

    # -----------------------------------------------------
    # Aggregate statistics
    # -----------------------------------------------------

    if statistics:

        mean_magnitude = float(
            np.mean(
                [
                    item["mean_magnitude"]
                    for item in statistics
                ]
            )
        )

        maximum_magnitude = float(
            np.max(
                [
                    item["max_magnitude"]
                    for item in statistics
                ]
            )
        )

        mean_horizontal_motion = float(
            np.mean(
                [
                    item["mean_horizontal_motion"]
                    for item in statistics
                ]
            )
        )

        mean_vertical_motion = float(
            np.mean(
                [
                    item["mean_vertical_motion"]
                    for item in statistics
                ]
            )
        )

    else:

        mean_magnitude = 0.0
        maximum_magnitude = 0.0
        mean_horizontal_motion = 0.0
        mean_vertical_motion = 0.0

    # -----------------------------------------------------
    # Output metadata
    # -----------------------------------------------------

    result = {
        "video_name": name,
        "input_file": str(input_path),
        "input_fps": float(fps),
        "input_frame_count": frame_count,
        "input_resolution": [
            original_width,
            original_height,
        ],
        "input_duration_seconds": float(duration),
        "processed_resolution": [
            PROCESS_WIDTH,
            PROCESS_HEIGHT,
        ],
        "frames_processed": frame_index - 1,
        "optical_flow_method": "Farneback dense optical flow",
        "mean_flow_magnitude": mean_magnitude,
        "maximum_flow_magnitude": maximum_magnitude,
        "mean_horizontal_motion": mean_horizontal_motion,
        "mean_vertical_motion": mean_vertical_motion,
        "hsv_output": str(hsv_output_path),
        "vector_output": str(vector_output_path),
        "sample_hsv_image": str(sample_hsv_path),
        "sample_vector_image": str(sample_vector_path),
    }

    metadata_path = (
        OUTPUT_DIR / f"{name}_optical_flow_metadata.json"
    )

    with open(
        metadata_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            result,
            file,
            indent=4,
        )

    # -----------------------------------------------------
    # Print summary
    # -----------------------------------------------------

    print()
    print(f"Completed: {name}")
    print(
        f"Mean flow magnitude: "
        f"{mean_magnitude:.4f}"
    )
    print(
        f"Maximum flow magnitude: "
        f"{maximum_magnitude:.4f}"
    )
    print(
        f"Mean horizontal motion: "
        f"{mean_horizontal_motion:.4f}"
    )
    print(
        f"Mean vertical motion: "
        f"{mean_vertical_motion:.4f}"
    )

    print()
    print("Saved:")
    print(f"  {hsv_output_path}")
    print(f"  {vector_output_path}")
    print(f"  {sample_hsv_path}")
    print(f"  {sample_vector_path}")
    print(f"  {metadata_path}")

    return result


def main():

    print()
    print("CSC 8830 - Module 6 Optical Flow")
    print("=" * 60)

    all_results = []

    for name, input_path in VIDEOS.items():

        result = process_video(
            name,
            input_path,
        )

        all_results.append(result)

    combined_metadata_path = (
        OUTPUT_DIR / "optical_flow_summary.json"
    )

    with open(
        combined_metadata_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            all_results,
            file,
            indent=4,
        )

    print()
    print("=" * 60)
    print("OPTICAL FLOW PROCESSING COMPLETE")
    print("=" * 60)
    print(
        f"Summary saved to: "
        f"{combined_metadata_path}"
    )


if __name__ == "__main__":
    main()
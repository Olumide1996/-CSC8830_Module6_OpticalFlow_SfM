"""
Extract fixed 30-second samples from the two Module 6 videos.

Selected experimental intervals:
    people_walking.mp4 : 13.0 s to 43.0 s
    cars_traffic.mp4   :  7.0 s to 37.0 s

The script:
1. Opens each original video.
2. Extracts the requested time interval.
3. Writes a new MP4 clip.
4. Saves representative frames for visual verification.
5. Prints the resulting clip metadata.

Run from the project root:

    python src\\extract_video_samples.py
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[1]

VIDEO_DIR = PROJECT_ROOT / "data" / "videos"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "optical_flow" / "video_samples"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


SAMPLES = [
    {
        "name": "people_walking",
        "input": VIDEO_DIR / "people_walking.mp4",
        "output": OUTPUT_DIR / "people_walking_30s.mp4",
        "start_seconds": 13.0,
        "end_seconds": 43.0,
    },
    {
        "name": "cars_traffic",
        "input": VIDEO_DIR / "cars_traffic.mp4",
        "output": OUTPUT_DIR / "cars_traffic_30s.mp4",
        "start_seconds": 7.0,
        "end_seconds": 37.0,
    },
]


def extract_segment(
    input_path: Path,
    output_path: Path,
    start_seconds: float,
    end_seconds: float,
) -> dict:
    """Extract a time interval from a video and save metadata."""

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input video does not exist: {input_path}"
        )

    if end_seconds <= start_seconds:
        raise ValueError(
            "end_seconds must be greater than start_seconds."
        )

    capture = cv2.VideoCapture(str(input_path))

    if not capture.isOpened():
        raise RuntimeError(
            f"Could not open video: {input_path}"
        )

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if fps <= 0:
        capture.release()
        raise RuntimeError(
            f"Invalid FPS reported by video: {input_path}"
        )

    duration_seconds = total_frames / fps

    if end_seconds > duration_seconds:
        capture.release()
        raise ValueError(
            f"Requested end time {end_seconds:.2f}s exceeds "
            f"video duration {duration_seconds:.2f}s."
        )

    start_frame = int(round(start_seconds * fps))
    end_frame = int(round(end_seconds * fps))

    capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        fps,
        (width, height),
    )

    if not writer.isOpened():
        capture.release()
        raise RuntimeError(
            f"Could not create output video: {output_path}"
        )

    saved_frames = 0

    representative_indices = {
        start_frame,
        start_frame + max(1, (end_frame - start_frame) // 2),
        max(start_frame, end_frame - 1),
    }

    representative_frames = {}

    while True:

        current_frame = int(
            capture.get(cv2.CAP_PROP_POS_FRAMES)
        )

        if current_frame >= end_frame:
            break

        success, frame = capture.read()

        if not success:
            break

        writer.write(frame)
        saved_frames += 1

        if current_frame in representative_indices:

            if current_frame == start_frame:
                label = "start"

            elif current_frame == max(
                start_frame,
                end_frame - 1,
            ):
                label = "end"

            else:
                label = "middle"

            representative_path = (
                OUTPUT_DIR
                / f"{input_path.stem}_{label}.jpg"
            )

            cv2.imwrite(
                str(representative_path),
                frame,
            )

            representative_frames[label] = (
                str(representative_path.relative_to(PROJECT_ROOT))
            )

    capture.release()
    writer.release()

    output_capture = cv2.VideoCapture(
        str(output_path)
    )

    if not output_capture.isOpened():
        raise RuntimeError(
            f"Created output video could not be reopened: "
            f"{output_path}"
        )

    output_fps = float(
        output_capture.get(cv2.CAP_PROP_FPS)
    )

    output_frames = int(
        output_capture.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    output_width = int(
        output_capture.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    output_height = int(
        output_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    output_capture.release()

    output_duration = (
        output_frames / output_fps
        if output_fps > 0
        else 0.0
    )

    metadata = {
        "input_file": str(
            input_path.relative_to(PROJECT_ROOT)
        ),
        "output_file": str(
            output_path.relative_to(PROJECT_ROOT)
        ),
        "source_fps": fps,
        "source_frame_count": total_frames,
        "source_resolution": [
            width,
            height,
        ],
        "source_duration_seconds": duration_seconds,
        "selected_start_seconds": start_seconds,
        "selected_end_seconds": end_seconds,
        "requested_duration_seconds": (
            end_seconds - start_seconds
        ),
        "start_frame": start_frame,
        "end_frame": end_frame,
        "saved_frame_count": saved_frames,
        "output_fps": output_fps,
        "output_frame_count": output_frames,
        "output_resolution": [
            output_width,
            output_height,
        ],
        "output_duration_seconds": output_duration,
        "representative_frames": representative_frames,
    }

    return metadata


def main() -> None:
    """Extract both experimental video segments."""

    all_metadata = {}

    print("\nModule 6 video extraction\n")
    print("=" * 60)

    for sample in SAMPLES:

        print(
            f"\nProcessing: {sample['name']}"
        )

        metadata = extract_segment(
            input_path=sample["input"],
            output_path=sample["output"],
            start_seconds=sample["start_seconds"],
            end_seconds=sample["end_seconds"],
        )

        all_metadata[sample["name"]] = metadata

        print(
            f"Input duration:  "
            f"{metadata['source_duration_seconds']:.2f} s"
        )

        print(
            f"Selected interval: "
            f"{metadata['selected_start_seconds']:.2f} - "
            f"{metadata['selected_end_seconds']:.2f} s"
        )

        print(
            f"Output duration: "
            f"{metadata['output_duration_seconds']:.2f} s"
        )

        print(
            f"Output frames: "
            f"{metadata['output_frame_count']}"
        )

        print(
            f"Output resolution: "
            f"{metadata['output_resolution'][0]} x "
            f"{metadata['output_resolution'][1]}"
        )

        print(
            f"Saved to: "
            f"{metadata['output_file']}"
        )

    metadata_path = OUTPUT_DIR / "sample_metadata.json"

    metadata_path.write_text(
        json.dumps(
            all_metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n" + "=" * 60)
    print("Video extraction completed.")
    print(
        f"Metadata saved to: "
        f"{metadata_path.relative_to(PROJECT_ROOT)}"
    )


if __name__ == "__main__":
    main()
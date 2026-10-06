"""
Four-view planar SFM visualization.

Inputs:
    data/sfm/view1.jpeg
    data/sfm/view2.jpeg
    data/sfm/view3.jpeg
    data/sfm/view4.jpeg

    outputs/sfm/relative_sfm_results.json
    outputs/sfm/a4_boundary_corners.json

Outputs:
    outputs/sfm/sfm_four_view_visualization.png
    outputs/sfm/sfm_planar_reconstruction.png

The visualization uses the calibrated-homography SFM results already
computed for the four planar A4 views.
"""

from pathlib import Path
import json

import cv2
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SFM_DIR = PROJECT_ROOT / "data" / "sfm"

RESULTS_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "sfm"
    / "relative_sfm_results.json"
)

CORNERS_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "sfm"
    / "a4_boundary_corners.json"
)

FOUR_VIEW_OUTPUT = (
    PROJECT_ROOT
    / "outputs"
    / "sfm"
    / "sfm_four_view_visualization.png"
)

PLANAR_OUTPUT = (
    PROJECT_ROOT
    / "outputs"
    / "sfm"
    / "sfm_planar_reconstruction.png"
)


IMAGE_FILES = [
    SFM_DIR / "view1.jpeg",
    SFM_DIR / "view2.jpeg",
    SFM_DIR / "view3.jpeg",
    SFM_DIR / "view4.jpeg",
]


# ============================================================
# LOAD DATA
# ============================================================

def load_json(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Required file was not found:\n{path}"
        )

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as file:
        return json.load(file)


def load_images():
    images = []

    for path in IMAGE_FILES:

        if not path.exists():
            raise FileNotFoundError(
                f"Required image was not found:\n{path}"
            )

        image = cv2.imread(
            str(path)
        )

        if image is None:
            raise RuntimeError(
                f"OpenCV could not read:\n{path}"
            )

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        images.append(image)

    return images


# ============================================================
# A4 CORNERS
# ============================================================

def load_a4_corners():
    data = load_json(
        CORNERS_FILE
    )

    # The selection script stores the coordinates under
    # "corners".
    if "corners" in data:
        corners_data = data["corners"]

    else:
        corners_data = data

    corners = np.asarray(
        [
            corners_data["top_left"],
            corners_data["top_right"],
            corners_data["bottom_right"],
            corners_data["bottom_left"],
        ],
        dtype=np.float32
    )

    return corners


# ============================================================
# HOMOGRAPHIES
# ============================================================

def load_pairwise_homographies(results):
    homographies = {}

    for pair_data in results["homography_pairs"]:

        pair = pair_data["pair"]

        H = np.asarray(
            pair_data["homography"],
            dtype=np.float64
        )

        homographies[pair] = H

    return homographies


def build_reference_to_view_homographies(
    homographies
):
    """
    Build:

        H12 : View 1 -> View 2
        H13 : View 1 -> View 3
        H14 : View 1 -> View 4

    using the pairwise homographies.

    Homography composition:

        H13 = H23 @ H12

        H14 = H34 @ H23 @ H12
    """

    H12 = homographies[
        "view1_to_view2"
    ]

    H23 = homographies[
        "view2_to_view3"
    ]

    H34 = homographies[
        "view3_to_view4"
    ]

    H13 = H23 @ H12

    H14 = H34 @ H23 @ H12

    return {
        1: np.eye(3),
        2: H12,
        3: H13,
        4: H14,
    }


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def detect_sift_features(
    image
):
    """
    Detect SIFT features for each image.
    """

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2GRAY
    )

    sift = cv2.SIFT_create(
        nfeatures=3000
    )

    keypoints, descriptors = sift.detectAndCompute(
        gray,
        None
    )

    return keypoints, descriptors


# ============================================================
# MATCHING
# ============================================================

def get_inlier_points(
    keypoints1,
    descriptors1,
    keypoints2,
    descriptors2
):
    """
    Match two views and estimate a robust homography.

    Returns the corresponding inlier points in both images.
    """

    matcher = cv2.BFMatcher(
        cv2.NORM_L2
    )

    matches = matcher.knnMatch(
        descriptors1,
        descriptors2,
        k=2
    )

    good_matches = []

    for pair in matches:

        if len(pair) != 2:
            continue

        first, second = pair

        if first.distance < 0.75 * second.distance:
            good_matches.append(
                first
            )

    if len(good_matches) < 4:
        raise RuntimeError(
            "Not enough good feature matches "
            "to estimate a homography."
        )

    source_points = np.float32(
        [
            keypoints1[
                match.queryIdx
            ].pt
            for match in good_matches
        ]
    )

    target_points = np.float32(
        [
            keypoints2[
                match.trainIdx
            ].pt
            for match in good_matches
        ]
    )

    H, mask = cv2.findHomography(
        source_points,
        target_points,
        cv2.RANSAC,
        5.0
    )

    if H is None or mask is None:
        raise RuntimeError(
            "Could not estimate homography."
        )

    mask = mask.ravel().astype(bool)

    return (
        source_points[mask],
        target_points[mask]
    )


# ============================================================
# FOUR-VIEW FEATURE POINT COLLECTION
# ============================================================

def collect_feature_points(
    images
):
    """
    Collect robust planar feature points for each view.

    View 1 uses inliers from View 1 -> View 2.
    View 2 uses inliers from both adjacent pairs.
    View 3 uses inliers from both adjacent pairs.
    View 4 uses inliers from View 3 -> View 4.
    """

    features = []

    for image in images:

        keypoints, descriptors = detect_sift_features(
            image
        )

        features.append(
            (
                keypoints,
                descriptors
            )
        )

    view_points = {
        1: [],
        2: [],
        3: [],
        4: [],
    }

    # --------------------------------------------------------
    # View 1 -> View 2
    # --------------------------------------------------------

    p1, p2 = get_inlier_points(
        features[0][0],
        features[0][1],
        features[1][0],
        features[1][1]
    )

    view_points[1].append(p1)
    view_points[2].append(p2)

    # --------------------------------------------------------
    # View 2 -> View 3
    # --------------------------------------------------------

    p2, p3 = get_inlier_points(
        features[1][0],
        features[1][1],
        features[2][0],
        features[2][1]
    )

    view_points[2].append(p2)
    view_points[3].append(p3)

    # --------------------------------------------------------
    # View 3 -> View 4
    # --------------------------------------------------------

    p3, p4 = get_inlier_points(
        features[2][0],
        features[2][1],
        features[3][0],
        features[3][1]
    )

    view_points[3].append(p3)
    view_points[4].append(p4)

    # Combine points from adjacent pairs.
    combined = {}

    for view in range(1, 5):

        if not view_points[view]:
            combined[view] = np.empty(
                (0, 2),
                dtype=np.float32
            )

            continue

        combined[view] = np.vstack(
            view_points[view]
        )

    return combined


# ============================================================
# DRAW A4 BOUNDARY
# ============================================================

def draw_boundary(
    ax,
    corners
):
    """
    Draw the manually validated A4 boundary.
    """

    closed = np.vstack(
        [
            corners,
            corners[0]
        ]
    )

    ax.plot(
        closed[:, 0],
        closed[:, 1],
        linewidth=2.5
    )


# ============================================================
# FOUR-VIEW FIGURE
# ============================================================

def create_four_view_figure(
    images,
    corners,
    feature_points
):
    """
    Create a 2x2 figure showing all four physical views.
    """

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(14, 12)
    )

    axes = axes.ravel()

    for index in range(4):

        view_number = index + 1

        ax = axes[index]

        ax.imshow(
            images[index]
        )

        # ----------------------------------------------------
        # Feature points
        # ----------------------------------------------------

        points = feature_points[
            view_number
        ]

        # Avoid excessive visual clutter.
        if len(points) > 250:

            selected_indices = np.linspace(
                0,
                len(points) - 1,
                250
            ).astype(int)

            display_points = points[
                selected_indices
            ]

        else:

            display_points = points

        if len(display_points) > 0:

            ax.scatter(
                display_points[:, 0],
                display_points[:, 1],
                s=5,
                alpha=0.45
            )

        # ----------------------------------------------------
        # A4 boundary
        # ----------------------------------------------------

        draw_boundary(
            ax,
            corners[index]
        )

        # ----------------------------------------------------
        # View label
        # ----------------------------------------------------

        ax.set_title(
            f"View {view_number}",
            fontsize=13
        )

        ax.axis(
            "off"
        )

    fig.suptitle(
        "Four-View Planar Structure-from-Motion Reconstruction",
        fontsize=16,
        y=0.98
    )

    fig.text(
        0.5,
        0.02,
        "A4 boundary: manually selected physical paper corners. "
        "Points: robust SIFT/RANSAC planar correspondences.",
        ha="center",
        fontsize=10
    )

    fig.tight_layout(
        rect=[
            0,
            0.04,
            1,
            0.95
        ]
    )

    fig.savefig(
        FOUR_VIEW_OUTPUT,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(fig)


# ============================================================
# PLANAR RECONSTRUCTION
# ============================================================

def create_planar_reconstruction(
    feature_points,
    reference_to_view
):
    """
    Transform feature points from each view into the
    View 1 reference image coordinate system.

    This creates a common planar reference representation.
    """

    reconstructed_points = []

    for view_number in range(1, 5):

        points = feature_points[
            view_number
        ]

        if len(points) == 0:
            continue

        H = reference_to_view[
            view_number
        ]

        H_inverse = np.linalg.inv(
            H
        )

        points_for_transform = points.reshape(
            -1,
            1,
            2
        ).astype(
            np.float32
        )

        reference_points = cv2.perspectiveTransform(
            points_for_transform,
            H_inverse.astype(
                np.float64
            )
        )

        reference_points = reference_points.reshape(
            -1,
            2
        )

        reconstructed_points.append(
            reference_points
        )

    if not reconstructed_points:

        return np.empty(
            (0, 2),
            dtype=np.float32
        )

    return np.vstack(
        reconstructed_points
    )


# ============================================================
# PLANAR RECONSTRUCTION FIGURE
# ============================================================

def create_planar_figure(
    reconstructed_points,
    reference_corners
):
    """
    Plot the reconstructed planar feature points in the
    View 1 reference coordinate system.
    """

    fig, ax = plt.subplots(
        figsize=(10, 8)
    )

    # --------------------------------------------------------
    # Points
    # --------------------------------------------------------

    if len(reconstructed_points) > 0:

        if len(reconstructed_points) > 1200:

            selected_indices = np.linspace(
                0,
                len(reconstructed_points) - 1,
                1200
            ).astype(int)

            display_points = reconstructed_points[
                selected_indices
            ]

        else:

            display_points = reconstructed_points

        ax.scatter(
            display_points[:, 0],
            display_points[:, 1],
            s=6,
            alpha=0.45,
            label="Reconstructed planar points"
        )

    # --------------------------------------------------------
    # A4 boundary
    # --------------------------------------------------------

    closed = np.vstack(
        [
            reference_corners,
            reference_corners[0]
        ]
    )

    ax.plot(
        closed[:, 0],
        closed[:, 1],
        linewidth=2.5,
        label="A4 boundary"
    )

    # --------------------------------------------------------
    # Corner labels
    # --------------------------------------------------------

    corner_names = [
        "TL",
        "TR",
        "BR",
        "BL"
    ]

    for point, name in zip(
        reference_corners,
        corner_names
    ):

        ax.annotate(
            name,
            (
                point[0],
                point[1]
            ),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=10
        )

    # --------------------------------------------------------
    # Formatting
    # --------------------------------------------------------

    ax.set_title(
        "Planar SFM Reconstruction in View 1 Reference Coordinates",
        fontsize=14
    )

    ax.set_xlabel(
        "Reference X (pixels)"
    )

    ax.set_ylabel(
        "Reference Y (pixels)"
    )

    ax.invert_yaxis()

    ax.set_aspect(
        "equal",
        adjustable="box"
    )

    ax.legend()

    ax.grid(
        True,
        alpha=0.25
    )

    fig.text(
        0.5,
        0.015,
        "All four views are mapped into the View 1 reference plane "
        "using the estimated pairwise homographies.",
        ha="center",
        fontsize=9
    )

    fig.tight_layout(
        rect=[
            0,
            0.04,
            1,
            1
        ]
    )

    fig.savefig(
        PLANAR_OUTPUT,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(fig)


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 60)
    print("FOUR-VIEW PLANAR SFM VISUALIZATION")
    print("=" * 60)
    print()

    # --------------------------------------------------------
    # Load results.
    # --------------------------------------------------------

    results = load_json(
        RESULTS_FILE
    )

    print(
        "Loaded SFM results."
    )

    # --------------------------------------------------------
    # Load images.
    # --------------------------------------------------------

    images = load_images()

    print(
        f"Loaded {len(images)} images."
    )

    # --------------------------------------------------------
    # Load A4 corners.
    # --------------------------------------------------------

    corners_view1 = load_a4_corners()

    print(
        "Loaded manually validated A4 corners."
    )

    # --------------------------------------------------------
    # The same physical boundary must be propagated through
    # the pairwise homographies.
    # --------------------------------------------------------

    homographies = load_pairwise_homographies(
        results
    )

    reference_to_view = (
        build_reference_to_view_homographies(
            homographies
        )
    )

    corners_by_view = {}

    corners_input = corners_view1.reshape(
        -1,
        1,
        2
    )

    for view_number in range(1, 5):

        H = reference_to_view[
            view_number
        ]

        transformed = cv2.perspectiveTransform(
            corners_input,
            H
        )

        corners_by_view[
            view_number
        ] = transformed.reshape(
            -1,
            2
        )

    # --------------------------------------------------------
    # Feature correspondences.
    # --------------------------------------------------------

    feature_points = collect_feature_points(
        images
    )

    for view_number in range(1, 5):

        print(
            f"View {view_number}: "
            f"{len(feature_points[view_number])} "
            "robust feature points"
        )

    # --------------------------------------------------------
    # Four-view visualization.
    # --------------------------------------------------------

    create_four_view_figure(
        images,
        [
            corners_by_view[1],
            corners_by_view[2],
            corners_by_view[3],
            corners_by_view[4],
        ],
        feature_points
    )

    print()
    print(
        f"Saved: {FOUR_VIEW_OUTPUT}"
    )

    # --------------------------------------------------------
    # Common planar reconstruction.
    # --------------------------------------------------------

    reconstructed_points = create_planar_reconstruction(
        feature_points,
        reference_to_view
    )

    print(
        f"Total reconstructed planar points: "
        f"{len(reconstructed_points)}"
    )

    create_planar_figure(
        reconstructed_points,
        corners_by_view[1]
    )

    print(
        f"Saved: {PLANAR_OUTPUT}"
    )

    print()
    print(
        "Four-view SFM visualization complete."
    )


if __name__ == "__main__":
    main()
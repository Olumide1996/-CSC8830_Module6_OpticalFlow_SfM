"""
Four-View Planar Structure from Motion

This script:
1. Loads four views of the same planar A4 object.
2. Loads the previously calculated camera calibration.
3. Detects SIFT features in all four views.
4. Matches features between consecutive views.
5. Estimates robust planar homographies using RANSAC.
6. Decomposes the calibrated homographies into candidate camera motions.
7. Builds a common planar reference coordinate system.
8. Detects the physical A4 sheet boundary in the reference image.
9. Saves feature matches, homography visualizations, reconstructed points,
   camera information, and mathematical results for the report.

The assignment allows a flat/2D planar object for the SFM example.
"""

from pathlib import Path
import json

import cv2
import numpy as np


# =====================================================================
# PROJECT PATHS
# =====================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data" / "sfm"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "sfm"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =====================================================================
# INPUT FILES
# =====================================================================

IMAGE_FILES = [
    DATA_DIR / "view1.jpeg",
    DATA_DIR / "view2.jpeg",
    DATA_DIR / "view3.jpeg",
    DATA_DIR / "view4.jpeg",
]

CALIBRATION_FILE = DATA_DIR / "camera_calibration.npz"


# =====================================================================
# PARAMETERS
# =====================================================================

SIFT_FEATURES = 3000
LOWE_RATIO = 0.75
RANSAC_THRESHOLD = 4.0
MIN_MATCHES = 12


# =====================================================================
# CALIBRATION
# =====================================================================

def load_calibration():
    """Load the existing camera calibration."""

    if not CALIBRATION_FILE.exists():
        raise FileNotFoundError(
            f"Calibration file not found:\n{CALIBRATION_FILE}"
        )

    calibration = np.load(CALIBRATION_FILE)

    camera_matrix = calibration["camera_matrix"].astype(np.float64)
    dist_coeffs = calibration["dist_coeffs"].astype(np.float64)
    image_size = calibration["image_size"]

    return camera_matrix, dist_coeffs, image_size


# =====================================================================
# IMAGE LOADING
# =====================================================================

def load_images():
    """Load all four SFM images."""

    images = []

    for image_path in IMAGE_FILES:

        if not image_path.exists():
            raise FileNotFoundError(
                f"Image not found:\n{image_path}"
            )

        image = cv2.imread(str(image_path))

        if image is None:
            raise RuntimeError(
                f"OpenCV could not read:\n{image_path}"
            )

        images.append(image)

    return images


# =====================================================================
# FEATURE DETECTION
# =====================================================================

def detect_features(images):
    """Detect SIFT features in all views."""

    sift = cv2.SIFT_create(
        nfeatures=SIFT_FEATURES
    )

    feature_data = []

    for index, image in enumerate(images, start=1):

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

        keypoints, descriptors = sift.detectAndCompute(
            gray,
            None
        )

        if descriptors is None or len(keypoints) < MIN_MATCHES:
            raise RuntimeError(
                f"Not enough features detected in View {index}."
            )

        feature_data.append(
            {
                "keypoints": keypoints,
                "descriptors": descriptors,
            }
        )

        print(
            f"View {index}: "
            f"{len(keypoints)} keypoints detected"
        )

    return feature_data


# =====================================================================
# FEATURE MATCHING
# =====================================================================

def match_features(descriptors1, descriptors2):
    """Match SIFT descriptors using FLANN and Lowe's ratio test."""

    index_params = dict(
        algorithm=1,
        trees=5
    )

    search_params = dict(
        checks=50
    )

    matcher = cv2.FlannBasedMatcher(
        index_params,
        search_params
    )

    raw_matches = matcher.knnMatch(
        descriptors1,
        descriptors2,
        k=2
    )

    good_matches = []

    for pair in raw_matches:

        if len(pair) != 2:
            continue

        first, second = pair

        if first.distance < LOWE_RATIO * second.distance:
            good_matches.append(first)

    return good_matches


def points_from_matches(
    keypoints1,
    keypoints2,
    matches
):
    """Convert feature matches into corresponding coordinates."""

    points1 = np.float32(
        [
            keypoints1[m.queryIdx].pt
            for m in matches
        ]
    )

    points2 = np.float32(
        [
            keypoints2[m.trainIdx].pt
            for m in matches
        ]
    )

    return points1, points2


# =====================================================================
# HOMOGRAPHY
# =====================================================================

def estimate_homography(points1, points2):
    """Estimate a robust homography using RANSAC."""

    H, mask = cv2.findHomography(
        points1,
        points2,
        method=cv2.RANSAC,
        ransacReprojThreshold=RANSAC_THRESHOLD
    )

    if H is None or mask is None:
        raise RuntimeError(
            "Homography estimation failed."
        )

    inlier_mask = mask.ravel().astype(bool)

    inlier_points1 = points1[inlier_mask]
    inlier_points2 = points2[inlier_mask]

    return (
        H,
        inlier_mask,
        inlier_points1,
        inlier_points2
    )


def compute_reprojection_error(
    H,
    points1,
    points2
):
    """Calculate mean Euclidean reprojection error."""

    points1_h = cv2.convertPointsToHomogeneous(
        points1
    ).reshape(-1, 3)

    projected = (
        H @ points1_h.T
    ).T

    projected = (
        projected[:, :2]
        /
        projected[:, 2:3]
    )

    errors = np.linalg.norm(
        projected - points2,
        axis=1
    )

    return float(
        np.mean(errors)
    )


# =====================================================================
# VISUALIZATION
# =====================================================================

def draw_match_visualization(
    image1,
    image2,
    keypoints1,
    keypoints2,
    matches,
    inlier_mask,
    output_path,
    title
):
    """Draw only RANSAC inlier feature correspondences."""

    inlier_matches = [
        match
        for match, is_inlier
        in zip(matches, inlier_mask)
        if is_inlier
    ]

    visualization = cv2.drawMatches(
        image1,
        keypoints1,
        image2,
        keypoints2,
        inlier_matches,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    cv2.putText(
        visualization,
        title,
        (30, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 0),
        2,
        cv2.LINE_AA
    )

    cv2.imwrite(
        str(output_path),
        visualization
    )


def save_homography_visualization(
    image1,
    image2,
    H,
    output_path,
    title
):
    """Warp image1 into image2 coordinates."""

    height, width = image2.shape[:2]

    warped = cv2.warpPerspective(
        image1,
        H,
        (width, height)
    )

    blended = cv2.addWeighted(
        image2,
        0.5,
        warped,
        0.5,
        0
    )

    cv2.putText(
        blended,
        title,
        (30, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 0),
        2,
        cv2.LINE_AA
    )

    cv2.imwrite(
        str(output_path),
        blended
    )


# =====================================================================
# HOMOGRAPHY DECOMPOSITION
# =====================================================================

def decompose_homography(
    H,
    camera_matrix
):
    """
    Decompose calibrated homography.

    For a planar scene:

        H = K [r1 r2 t] K^-1

    up to scale.
    """

    (
        solution_count,
        rotations,
        translations,
        normals
    ) = cv2.decomposeHomographyMat(
        H,
        camera_matrix
    )

    return (
        rotations,
        translations,
        normals
    )


# =====================================================================
# POINT TRANSFORMATION
# =====================================================================

def transform_points(
    H,
    points
):
    """Apply a homography to a set of 2D points."""

    points = np.asarray(
        points,
        dtype=np.float32
    ).reshape(-1, 1, 2)

    transformed = cv2.perspectiveTransform(
        points,
        H
    )

    return transformed.reshape(-1, 2)


# =====================================================================
# A4 BOUNDARY DETECTION
# =====================================================================

def order_corners(points):
    """
    Order four corner points as:

        top-left
        top-right
        bottom-right
        bottom-left
    """

    points = np.asarray(
        points,
        dtype=np.float32
    )

    ordered = np.zeros(
        (4, 2),
        dtype=np.float32
    )

    sums = points.sum(axis=1)
    differences = np.diff(
        points,
        axis=1
    ).ravel()

    ordered[0] = points[np.argmin(sums)]
    ordered[2] = points[np.argmax(sums)]

    ordered[1] = points[np.argmin(differences)]
    ordered[3] = points[np.argmax(differences)]

    return ordered


def detect_a4_boundary(image):
    """
    Detect the physical rectangular paper boundary.

    The detector searches for large convex quadrilaterals and
    favors shapes whose aspect ratio is compatible with A4 paper.

    A4 aspect ratio:

        210 / 297 = 0.7071
    """

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    blurred = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    edges = cv2.Canny(
        blurred,
        50,
        150
    )

    # Close small gaps along the paper boundary.
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (7, 7)
    )

    edges = cv2.morphologyEx(
        edges,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2
    )

    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE
    )

    image_height, image_width = gray.shape

    image_area = (
        image_height * image_width
    )

    candidates = []

    for contour in contours:

        area = cv2.contourArea(
            contour
        )

        # The sheet should occupy a meaningful portion of the image.
        if area < image_area * 0.10:
            continue

        perimeter = cv2.arcLength(
            contour,
            True
        )

        approximation = cv2.approxPolyDP(
            contour,
            0.02 * perimeter,
            True
        )

        if len(approximation) != 4:
            continue

        points = approximation.reshape(
            4,
            2
        ).astype(np.float32)

        if not cv2.isContourConvex(
            points.astype(np.int32)
        ):
            continue

        ordered = order_corners(
            points
        )

        # Calculate side lengths.
        top = np.linalg.norm(
            ordered[1] - ordered[0]
        )

        bottom = np.linalg.norm(
            ordered[2] - ordered[3]
        )

        left = np.linalg.norm(
            ordered[3] - ordered[0]
        )

        right = np.linalg.norm(
            ordered[2] - ordered[1]
        )

        average_width = (
            top + bottom
        ) / 2.0

        average_height = (
            left + right
        ) / 2.0

        if average_width <= 0 or average_height <= 0:
            continue

        ratio = (
            min(
                average_width,
                average_height
            )
            /
            max(
                average_width,
                average_height
            )
        )

        # A4 ratio is approximately 0.707.
        # Perspective can change this, so use a generous range.
        ratio_difference = abs(
            ratio - 0.7071
        )

        # Reject extremely narrow shapes.
        if ratio < 0.45:
            continue

        # Score:
        # - large area is good
        # - A4-like aspect ratio is good
        area_score = area / image_area

        ratio_score = max(
            0.0,
            1.0 - ratio_difference
        )

        score = (
            2.0 * area_score
            +
            ratio_score
        )

        candidates.append(
            (
                score,
                area,
                ordered
            )
        )

    if not candidates:
        return None

    candidates.sort(
        key=lambda item: item[0],
        reverse=True
    )

    best = candidates[0]

    return best[2]


def draw_a4_boundary(
    image,
    corners,
    output_path
):
    """Draw detected A4 corners and boundary."""

    result = image.copy()

    if corners is not None:

        corners_int = np.round(
            corners
        ).astype(np.int32)

        cv2.polylines(
            result,
            [corners_int],
            True,
            (0, 255, 0),
            5
        )

        labels = [
            "TL",
            "TR",
            "BR",
            "BL"
        ]

        for point, label in zip(
            corners_int,
            labels
        ):

            x, y = point

            cv2.circle(
                result,
                (int(x), int(y)),
                9,
                (0, 0, 255),
                -1
            )

            cv2.putText(
                result,
                label,
                (
                    int(x) + 10,
                    int(y) - 10
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
                cv2.LINE_AA
            )

    cv2.putText(
        result,
        "Detected A4 Planar Object Boundary",
        (30, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 0),
        2,
        cv2.LINE_AA
    )

    cv2.imwrite(
        str(output_path),
        result
    )


# =====================================================================
# MAIN
# =====================================================================

def main():

    print("=" * 70)
    print("CSC 8830 - FOUR-VIEW PLANAR SFM RECONSTRUCTION")
    print("=" * 70)

    # -----------------------------------------------------------------
    # 1. CALIBRATION
    # -----------------------------------------------------------------

    print("\n[1/8] Loading camera calibration...")

    (
        camera_matrix,
        dist_coeffs,
        image_size
    ) = load_calibration()

    print("\nCamera matrix:")
    print(camera_matrix)

    print("\nDistortion coefficients:")
    print(dist_coeffs)

    print(
        "\nCalibration image size:",
        image_size.tolist()
    )

    # -----------------------------------------------------------------
    # 2. IMAGES
    # -----------------------------------------------------------------

    print("\n[2/8] Loading four SFM images...")

    images = load_images()

    expected_width = int(
        image_size[0]
    )

    expected_height = int(
        image_size[1]
    )

    for index, image in enumerate(
        images,
        start=1
    ):

        height, width = image.shape[:2]

        print(
            f"View {index}: "
            f"{width} x {height}"
        )

        if (
            width != expected_width
            or
            height != expected_height
        ):
            raise RuntimeError(
                f"View {index} size does not match calibration."
            )

    # -----------------------------------------------------------------
    # 3. FEATURES
    # -----------------------------------------------------------------

    print("\n[3/8] Detecting SIFT features...")

    feature_data = detect_features(
        images
    )

    # -----------------------------------------------------------------
    # 4. HOMOGRAPHIES
    # -----------------------------------------------------------------

    print(
        "\n[4/8] Estimating pairwise homographies..."
    )

    pair_results = []

    for i in range(3):

        j = i + 1

        print(
            f"\nMatching View {i + 1} -> View {j + 1}"
        )

        keypoints1 = feature_data[i]["keypoints"]
        descriptors1 = feature_data[i]["descriptors"]

        keypoints2 = feature_data[j]["keypoints"]
        descriptors2 = feature_data[j]["descriptors"]

        matches = match_features(
            descriptors1,
            descriptors2
        )

        print(
            f"Good matches: {len(matches)}"
        )

        if len(matches) < MIN_MATCHES:

            raise RuntimeError(
                f"Not enough reliable matches between "
                f"View {i + 1} and View {j + 1}."
            )

        points1, points2 = points_from_matches(
            keypoints1,
            keypoints2,
            matches
        )

        (
            H,
            inlier_mask,
            inlier_points1,
            inlier_points2
        ) = estimate_homography(
            points1,
            points2
        )

        inlier_count = int(
            np.sum(inlier_mask)
        )

        inlier_ratio = (
            inlier_count / len(matches)
        )

        reprojection_error = compute_reprojection_error(
            H,
            inlier_points1,
            inlier_points2
        )

        print(
            f"Inliers: "
            f"{inlier_count}/{len(matches)} "
            f"({inlier_ratio * 100:.2f}%)"
        )

        print(
            f"Mean reprojection error: "
            f"{reprojection_error:.4f} pixels"
        )

        print("\nHomography:")
        print(H)

        # -------------------------------------------------------------
        # SAVE HOMOGRAPHY
        # -------------------------------------------------------------

        homography_path = (
            OUTPUT_DIR
            /
            f"homography_view{i + 1}_to_view{j + 1}.json"
        )

        with open(
            homography_path,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                {
                    "source_view": i + 1,
                    "target_view": j + 1,
                    "homography": H.tolist(),
                    "total_matches": len(matches),
                    "inliers": inlier_count,
                    "inlier_ratio": inlier_ratio,
                    "mean_reprojection_error_pixels":
                        reprojection_error,
                },
                file,
                indent=4
            )

        # -------------------------------------------------------------
        # MATCH VISUALIZATION
        # -------------------------------------------------------------

        match_output = (
            OUTPUT_DIR
            /
            f"matches_view{i + 1}_view{j + 1}.jpg"
        )

        draw_match_visualization(
            images[i],
            images[j],
            keypoints1,
            keypoints2,
            matches,
            inlier_mask,
            match_output,
            (
                f"View {i + 1} -> View {j + 1} | "
                f"{inlier_count} inliers"
            )
        )

        # -------------------------------------------------------------
        # HOMOGRAPHY WARP
        # -------------------------------------------------------------

        warp_output = (
            OUTPUT_DIR
            /
            f"homography_warp_view{i + 1}_view{j + 1}.jpg"
        )

        save_homography_visualization(
            images[i],
            images[j],
            H,
            warp_output,
            (
                f"Planar Homography "
                f"View {i + 1} -> View {j + 1}"
            )
        )

        # -------------------------------------------------------------
        # HOMOGRAPHY DECOMPOSITION
        # -------------------------------------------------------------

        (
            rotations,
            translations,
            normals
        ) = decompose_homography(
            H,
            camera_matrix
        )

        decomposition_path = (
            OUTPUT_DIR
            /
            f"homography_decomposition_view"
            f"{i + 1}_view{j + 1}.json"
        )

        decomposition_data = []

        for solution_index in range(
            len(rotations)
        ):

            decomposition_data.append(
                {
                    "solution":
                        solution_index + 1,
                    "rotation":
                        rotations[
                            solution_index
                        ].tolist(),
                    "translation_direction":
                        translations[
                            solution_index
                        ].ravel().tolist(),
                    "plane_normal":
                        normals[
                            solution_index
                        ].ravel().tolist(),
                }
            )

        with open(
            decomposition_path,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                decomposition_data,
                file,
                indent=4
            )

        pair_results.append(
            {
                "source_view":
                    i + 1,
                "target_view":
                    j + 1,
                "homography":
                    H,
                "matches":
                    matches,
                "points_source":
                    points1,
                "points_target":
                    points2,
                "inlier_mask":
                    inlier_mask,
                "inlier_points_source":
                    inlier_points1,
                "inlier_points_target":
                    inlier_points2,
                "inlier_count":
                    inlier_count,
                "inlier_ratio":
                    inlier_ratio,
                "reprojection_error":
                    reprojection_error,
                "rotations":
                    rotations,
                "translations":
                    translations,
                "normals":
                    normals,
            }
        )

    # -----------------------------------------------------------------
    # 5. COMMON PLANAR REFERENCE
    # -----------------------------------------------------------------

    print(
        "\n[5/8] Building common planar reference..."
    )

    H12 = pair_results[0]["homography"]
    H23 = pair_results[1]["homography"]
    H34 = pair_results[2]["homography"]

    # Chain the transformations:
    #
    # H13 = H23 H12
    # H14 = H34 H23 H12

    H13 = H23 @ H12
    H14 = H34 @ H23 @ H12

    # Inverse transformations bring later views
    # back into View 1 coordinates.

    H21 = np.linalg.inv(H12)
    H31 = np.linalg.inv(H13)
    H41 = np.linalg.inv(H14)

    # -----------------------------------------------------------------
    # 6. RECONSTRUCT PLANAR POINTS
    # -----------------------------------------------------------------

    print(
        "\n[6/8] Reconstructing common planar points..."
    )

    reference_points = []

    # View 1 points
    view1_points = pair_results[0][
        "inlier_points_source"
    ]

    reference_points.extend(
        view1_points.tolist()
    )

    # View 2 -> View 1
    view2_points = pair_results[0][
        "inlier_points_target"
    ]

    view2_to_reference = transform_points(
        H21,
        view2_points
    )

    reference_points.extend(
        view2_to_reference.tolist()
    )

    # View 3 -> View 1
    view3_points = pair_results[1][
        "inlier_points_target"
    ]

    view3_to_reference = transform_points(
        H31,
        view3_points
    )

    reference_points.extend(
        view3_to_reference.tolist()
    )

    # View 4 -> View 1
    view4_points = pair_results[2][
        "inlier_points_target"
    ]

    view4_to_reference = transform_points(
        H41,
        view4_points
    )

    reference_points.extend(
        view4_to_reference.tolist()
    )

    reference_points = np.asarray(
        reference_points,
        dtype=np.float32
    )

    print(
        "Total reconstructed planar points:",
        len(reference_points)
    )

    # -----------------------------------------------------------------
    # 7. DETECT PHYSICAL A4 BOUNDARY
    # -----------------------------------------------------------------

    print(
        "\n[7/8] Detecting physical A4 planar boundary..."
    )

    a4_corners = detect_a4_boundary(
        images[0]
    )

    if a4_corners is None:

        print(
            "WARNING: Automatic A4 boundary detection failed."
        )

        print(
            "The feature-based SFM reconstruction is still valid."
        )

    else:

        print("\nDetected A4 corners:")

        corner_names = [
            "Top-left",
            "Top-right",
            "Bottom-right",
            "Bottom-left"
        ]

        for name, point in zip(
            corner_names,
            a4_corners
        ):

            print(
                f"{name}: "
                f"x={point[0]:.2f}, "
                f"y={point[1]:.2f}"
            )

        boundary_output = (
            OUTPUT_DIR
            /
            "reconstructed_planar_boundary.jpg"
        )

        draw_a4_boundary(
            images[0],
            a4_corners,
            boundary_output
        )

        # -------------------------------------------------------------
        # Save the four corner coordinates.
        # -------------------------------------------------------------

        corners_output = (
            OUTPUT_DIR
            /
            "a4_boundary_corners.json"
        )

        with open(
            corners_output,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                {
                    "coordinate_system":
                        "View 1 image pixels",
                    "corners": {
                        name: [
                            float(point[0]),
                            float(point[1])
                        ]
                        for name, point in zip(
                            corner_names,
                            a4_corners
                        )
                    }
                },
                file,
                indent=4
            )

        # -------------------------------------------------------------
        # Save A4 plane dimensions.
        # -------------------------------------------------------------

        top_width = np.linalg.norm(
            a4_corners[1]
            -
            a4_corners[0]
        )

        bottom_width = np.linalg.norm(
            a4_corners[2]
            -
            a4_corners[3]
        )

        left_height = np.linalg.norm(
            a4_corners[3]
            -
            a4_corners[0]
        )

        right_height = np.linalg.norm(
            a4_corners[2]
            -
            a4_corners[1]
        )

        average_width = (
            top_width + bottom_width
        ) / 2.0

        average_height = (
            left_height + right_height
        ) / 2.0

        measured_ratio = (
            min(
                average_width,
                average_height
            )
            /
            max(
                average_width,
                average_height
            )
        )

        print(
            f"\nEstimated image-plane width: "
            f"{average_width:.2f} pixels"
        )

        print(
            f"Estimated image-plane height: "
            f"{average_height:.2f} pixels"
        )

        print(
            f"Estimated width/height ratio: "
            f"{measured_ratio:.4f}"
        )

    # -----------------------------------------------------------------
    # SAVE PLANAR POINT CLOUD
    # -----------------------------------------------------------------

    points_output = (
        OUTPUT_DIR
        /
        "reconstructed_planar_points.csv"
    )

    np.savetxt(
        points_output,
        reference_points,
        delimiter=",",
        header="x_reference,y_reference",
        comments=""
    )

    # -----------------------------------------------------------------
    # 8. SUMMARY
    # -----------------------------------------------------------------

    print(
        "\n[8/8] Saving final SFM summary..."
    )

    summary = {
        "method":
            "Four-view planar structure from motion "
            "using SIFT feature matching, RANSAC "
            "homographies, calibrated homography "
            "decomposition, and planar reconstruction",

        "reference_view":
            1,

        "number_of_views":
            4,

        "image_size": {
            "width":
                int(image_size[0]),
            "height":
                int(image_size[1]),
        },

        "camera_matrix":
            camera_matrix.tolist(),

        "distortion_coefficients":
            dist_coeffs.tolist(),

        "reconstructed_planar_points":
            int(len(reference_points)),

        "a4_boundary_detected":
            a4_corners is not None,

        "pairwise_results":
            [],
    }

    if a4_corners is not None:

        summary["a4_boundary_corners"] = {
            name: [
                float(point[0]),
                float(point[1])
            ]
            for name, point in zip(
                corner_names,
                a4_corners
            )
        }

        summary["a4_boundary_geometry"] = {
            "average_width_pixels":
                float(average_width),
            "average_height_pixels":
                float(average_height),
            "measured_ratio":
                float(measured_ratio),
            "expected_a4_ratio":
                210.0 / 297.0,
        }

    for result in pair_results:

        summary["pairwise_results"].append(
            {
                "source_view":
                    result["source_view"],

                "target_view":
                    result["target_view"],

                "matches":
                    len(result["matches"]),

                "inliers":
                    result["inlier_count"],

                "inlier_ratio":
                    result["inlier_ratio"],

                "mean_reprojection_error_pixels":
                    result["reprojection_error"],

                "homography":
                    result["homography"].tolist(),
            }
        )

    summary_output = (
        OUTPUT_DIR
        /
        "sfm_summary.json"
    )

    with open(
        summary_output,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            summary,
            file,
            indent=4
        )

    # -----------------------------------------------------------------
    # FINAL TERMINAL OUTPUT
    # -----------------------------------------------------------------

    print("\n" + "=" * 70)
    print("SFM RECONSTRUCTION COMPLETE")
    print("=" * 70)

    for result in pair_results:

        print(
            f"View {result['source_view']} -> "
            f"View {result['target_view']}: "
            f"{result['inlier_count']} inliers / "
            f"{len(result['matches'])} matches "
            f"("
            f"{result['inlier_ratio'] * 100:.2f}%"
            f"), "
            f"error = "
            f"{result['reprojection_error']:.4f} px"
        )

    print(
        f"\nReconstructed planar points: "
        f"{len(reference_points)}"
    )

    if a4_corners is not None:

        print(
            "A4 boundary detection: SUCCESS"
        )

    else:

        print(
            "A4 boundary detection: FAILED"
        )

    print(
        "\nResults saved to:"
    )

    print(
        OUTPUT_DIR
    )

    print("=" * 70)


# =====================================================================
# PROGRAM ENTRY POINT
# =====================================================================

if __name__ == "__main__":
    main()
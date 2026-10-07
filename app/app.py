import streamlit as st
from pathlib import Path


# ------------------------------------------------------------
# Page configuration
# ------------------------------------------------------------

st.set_page_config(
    page_title="CSc 8830 - Assignment 6",
    page_icon="CV",
    layout="wide",
)


# ------------------------------------------------------------
# Project paths
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
OPTICAL_FLOW_DIR = OUTPUTS_DIR / "optical_flow"
TRACKING_DIR = OUTPUTS_DIR / "tracking"
SFM_DIR = OUTPUTS_DIR / "sfm"
DATA_SFM_DIR = PROJECT_ROOT / "data" / "sfm"


# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------

def show_image(image_path, caption=None, width=700):
    """Display an image at a controlled size if it exists."""
    if image_path.exists():
        st.image(
            str(image_path),
            caption=caption,
            width=width,
        )
    else:
        st.warning(f"File not found: {image_path}")


def show_video(video_path):
    """Display a video if it exists."""
    if video_path.exists():
        st.video(str(video_path))
    else:
        st.warning(f"Video not found: {video_path}")


# ------------------------------------------------------------
# Header
# ------------------------------------------------------------

st.title("CSc 8830 — Assignment 6")

st.subheader("Optical Flow & Structure from Motion")

st.write(
    """
    This web application presents the main experimental results from
    Assignment 6, including optical flow, motion tracking,
    bilinear interpolation, and planar Structure from Motion.
    """
)

st.divider()


# ============================================================
# PART A — OPTICAL FLOW
# ============================================================

st.header("Part A — Optical Flow and Motion Tracking")

st.write(
    """
    Two 30-second video samples were analyzed using dense optical flow.
    The assignment requires optical-flow visualization, motion-tracking
    validation using consecutive frames, and a bilinear interpolation example.
    """
)


# ------------------------------------------------------------
# Optical Flow Results
# ------------------------------------------------------------

st.subheader("Optical Flow Results")

flow_col1, flow_col2 = st.columns(2)

with flow_col1:

    st.markdown("### People Walking")

    st.write(
        "Dense Farneback optical flow was computed for the "
        "30-second people-walking sequence."
    )

    people_flow_video = (
        OPTICAL_FLOW_DIR
        / "people_walking_optical_flow_hsv_web.mp4"
    )

    show_video(people_flow_video)

    st.metric(
        "Mean Flow Magnitude",
        "2.2337 px",
    )

    st.metric(
        "Maximum Flow Magnitude",
        "181.6555 px",
    )


with flow_col2:

    st.markdown("### Cars in Traffic")

    st.write(
        "Dense Farneback optical flow was computed for the "
        "30-second traffic sequence."
    )

    cars_flow_video = (
        OPTICAL_FLOW_DIR
        / "cars_traffic_optical_flow_hsv_web.mp4"
    )

    show_video(cars_flow_video)

    st.metric(
        "Mean Flow Magnitude",
        "0.3255 px",
    )

    st.metric(
        "Maximum Flow Magnitude",
        "24.3862 px",
    )


# ------------------------------------------------------------
# Optical Flow Interpretation
# ------------------------------------------------------------

st.subheader("What Information Does Optical Flow Provide?")

st.write(
    """
    Optical flow represents the apparent motion of image points between
    consecutive frames. It provides information about the direction and
    magnitude of image motion and can be used to understand movement of
    objects and changes across the scene.
    """
)


# ------------------------------------------------------------
# Motion Tracking Validation
# ------------------------------------------------------------

st.subheader("Motion Tracking Validation")

st.write(
    """
    Lucas-Kanade feature tracking was compared with predictions from the
    dense Farneback optical-flow field. A tracking error of 1 pixel or less
    was treated as an inlier.
    """
)

tracking_col1, tracking_col2 = st.columns(2)

with tracking_col1:

    st.markdown("### People Walking")

    people_tracking_image = (
        TRACKING_DIR
        / "people_walking_tracking_comparison.png"
    )

    show_image(
        people_tracking_image,
        caption="Tracking comparison using consecutive frames",
        width=500,
    )

    st.write("Points tracked: 20/20")
    st.write("Inliers: 14/20")
    st.write("Inlier rate: 70%")
    st.write("Robust mean error: 0.1793 px")


with tracking_col2:

    st.markdown("### Cars in Traffic")

    cars_tracking_image = (
        TRACKING_DIR
        / "cars_traffic_tracking_comparison.png"
    )

    show_image(
        cars_tracking_image,
        caption="Tracking comparison using consecutive frames",
        width=500,
    )

    st.write("Points tracked: 20/20")
    st.write("Inliers: 20/20")
    st.write("Inlier rate: 100%")
    st.write("Robust mean error: 0.0480 px")


# ------------------------------------------------------------
# Bilinear Interpolation
# ------------------------------------------------------------

st.subheader("Bilinear Interpolation")

st.write(
    """
    Bilinear interpolation was used to estimate the optical-flow vector
    at a subpixel location.
    """
)

bilinear_col1, bilinear_col2 = st.columns(2)

with bilinear_col1:

    st.markdown("#### Starting Location")

    st.latex(
        r"(x,y)=(695.35,\ 354.62)"
    )

    st.markdown("#### Interpolated Flow")

    st.latex(
        r"u=0.522021"
    )

    st.latex(
        r"v=1.264873"
    )


with bilinear_col2:

    st.markdown("#### Predicted Location")

    st.latex(
        r"x'=695.872021"
    )

    st.latex(
        r"y'=355.884873"
    )

st.info(
    "The predicted position is obtained by adding the interpolated "
    "flow vector to the starting image location."
)


# ============================================================
# PART B — STRUCTURE FROM MOTION
# ============================================================

st.divider()

st.header("Part B — Structure from Motion")

st.write(
    """
    A planar Structure from Motion example was constructed using four
    photographs of the same A4 object captured from different viewpoints.
    """
)


# ------------------------------------------------------------
# Four Input Views
# ------------------------------------------------------------

st.subheader("Four Camera Views")

view_col1, view_col2 = st.columns(2)

with view_col1:

    show_image(
        DATA_SFM_DIR / "view1.jpeg",
        caption="View 1",
        width=480,
    )

    show_image(
        DATA_SFM_DIR / "view3.jpeg",
        caption="View 3",
        width=480,
    )


with view_col2:

    show_image(
        DATA_SFM_DIR / "view2.jpeg",
        caption="View 2",
        width=480,
    )

    show_image(
        DATA_SFM_DIR / "view4.jpeg",
        caption="View 4",
        width=480,
    )


# ------------------------------------------------------------
# Camera Calibration
# ------------------------------------------------------------

st.subheader("Camera Calibration")

st.write(
    "Camera matrix used for the reconstruction:"
)

st.latex(
    r"""
    K=
    \begin{bmatrix}
    797.626 & 0 & 386.727\\
    0 & 797.129 & 535.011\\
    0 & 0 & 1
    \end{bmatrix}
    """
)

cal_col1, cal_col2 = st.columns(2)

with cal_col1:

    st.metric(
        "Image Width",
        "810 px",
    )


with cal_col2:

    st.metric(
        "Image Height",
        "1080 px",
    )


st.metric(
    "Mean Calibration Reprojection Error",
    "0.6274 px",
)


# ------------------------------------------------------------
# Homography Results
# ------------------------------------------------------------

st.subheader("Homography Results")

st.table(
    {
        "View Transition": [
            "View 1 → View 2",
            "View 2 → View 3",
            "View 3 → View 4",
        ],
        "Good Matches": [
            1268,
            1136,
            984,
        ],
        "Inliers": [
            1107,
            995,
            870,
        ],
        "Inlier Rate": [
            "87.30%",
            "87.59%",
            "88.41%",
        ],
        "Mean Reprojection Error": [
            "0.536 px",
            "0.910 px",
            "0.873 px",
        ],
    }
)


# ------------------------------------------------------------
# Four-View Reconstruction
# ------------------------------------------------------------

st.subheader("Four-View Reconstruction")

sfm_four_view = (
    SFM_DIR
    / "sfm_four_view_visualization.png"
)

show_image(
    sfm_four_view,
    caption="Four-view planar Structure from Motion reconstruction",
    width=700,
)


# ------------------------------------------------------------
# Relative Camera Configuration
# ------------------------------------------------------------

st.subheader("Relative Camera Configuration")

camera_configuration = (
    SFM_DIR
    / "camera_configuration_3d.png"
)

show_image(
    camera_configuration,
    caption="Reconstructed relative camera configuration",
    width=700,
)


# ------------------------------------------------------------
# Planar Reconstruction
# ------------------------------------------------------------

st.subheader("Planar Reconstruction")

planar_reconstruction = (
    SFM_DIR
    / "sfm_planar_reconstruction.png"
)

show_image(
    planar_reconstruction,
    caption="Planar SFM reconstruction",
    width=700,
)


# ------------------------------------------------------------
# Relative Camera Positions
# ------------------------------------------------------------

st.subheader("Relative Camera Positions")

st.write(
    "Camera positions are expressed in relative SFM units."
)

st.table(
    {
        "View": [
            "View 1",
            "View 2",
            "View 3",
            "View 4",
        ],
        "X": [
            "0.0000",
            "0.0131",
            "0.4018",
            "-0.3993",
        ],
        "Y": [
            "0.0000",
            "0.0115",
            "0.2314",
            "0.2451",
        ],
        "Z": [
            "0.0000",
            "-0.2807",
            "-0.1566",
            "-0.2694",
        ],
    }
)

st.caption(
    "Translation scale is arbitrary for monocular planar SFM, so the "
    "camera positions are reported in relative SFM units rather than millimeters."
)


# ------------------------------------------------------------
# Homography Decomposition
# ------------------------------------------------------------

st.subheader("Homography Decomposition")

st.table(
    {
        "View Transition": [
            "View 1 → View 2",
            "View 2 → View 3",
            "View 3 → View 4",
        ],
        "Selected Solution": [
            1,
            4,
            1,
        ],
        "Positive-Depth Fraction": [
            "100%",
            "100%",
            "100%",
        ],
    }
)


# ------------------------------------------------------------
# Repository
# ------------------------------------------------------------

st.divider()

st.subheader("Project Repository")

st.link_button(
    "Open Assignment 6 GitHub Repository",
    "https://github.com/Olumide1996/-CSC8830_Module6_OpticalFlow_SfM",
)

st.info(
    "The complete source code, mathematical work, experimental data, "
    "and additional results are available in the GitHub repository."
)


# ------------------------------------------------------------
# Footer
# ------------------------------------------------------------

st.divider()

st.caption(
    "CSc 8830 — Computer Vision | Assignment 6 | Fall 2026"
)
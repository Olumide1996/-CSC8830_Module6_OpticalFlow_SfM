from pathlib import Path
import json

import numpy as np
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RESULTS_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "sfm"
    / "relative_sfm_results.json"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "sfm"
    / "camera_configuration_3d.png"
)


def load_results():
    if not RESULTS_FILE.exists():
        raise FileNotFoundError(
            f"Could not find:\n{RESULTS_FILE}"
        )

    with open(
        RESULTS_FILE,
        "r",
        encoding="utf-8"
    ) as file:
        return json.load(file)


def draw_camera(
    ax,
    center,
    rotation,
    label,
    scale=0.08
):
    """
    Draw one camera center and its orientation axes.
    """

    center = np.asarray(
        center,
        dtype=float
    )

    rotation = np.asarray(
        rotation,
        dtype=float
    )

    # The columns of R represent the camera's
    # local X, Y, and Z axes in the reference frame.
    x_axis = rotation[:, 0]
    y_axis = rotation[:, 1]
    z_axis = rotation[:, 2]

    # Camera center.
    ax.scatter(
        center[0],
        center[1],
        center[2],
        s=70,
        marker="o"
    )

    # Camera X axis.
    ax.quiver(
        center[0],
        center[1],
        center[2],
        x_axis[0],
        x_axis[1],
        x_axis[2],
        length=scale,
        normalize=True
    )

    # Camera Y axis.
    ax.quiver(
        center[0],
        center[1],
        center[2],
        y_axis[0],
        y_axis[1],
        y_axis[2],
        length=scale,
        normalize=True
    )

    # Camera Z axis.
    ax.quiver(
        center[0],
        center[1],
        center[2],
        z_axis[0],
        z_axis[1],
        z_axis[2],
        length=scale,
        normalize=True
    )

    # Label.
    ax.text(
        center[0],
        center[1],
        center[2],
        f"  {label}",
        fontsize=10
    )


def make_plot(results):

    camera_poses = results["camera_poses"]
    decompositions = results["decomposition"]

    # ---------------------------------------------------------
    # Create mapping:
    #
    # View 1 -> identity rotation
    # View 2 -> selected decomposition for View 1 -> View 2
    # View 3 -> selected decomposition for View 2 -> View 3
    # View 4 -> selected decomposition for View 3 -> View 4
    #
    # ---------------------------------------------------------

    rotations = {}

    rotations[1] = np.eye(3)

    for decomposition in decompositions:

        pair = decomposition["pair"]

        rotation = np.asarray(
            decomposition["rotation"],
            dtype=float
        )

        if pair == "view1_to_view2":
            rotations[2] = rotation

        elif pair == "view2_to_view3":
            rotations[3] = rotation

        elif pair == "view3_to_view4":
            rotations[4] = rotation

    # ---------------------------------------------------------
    # Extract camera centers.
    # ---------------------------------------------------------

    centers = []

    for pose in camera_poses:

        view_number = int(
            pose["view"]
        )

        center = np.asarray(
            pose["camera_center_sfm_units"],
            dtype=float
        )

        centers.append(center)

    centers = np.asarray(
        centers,
        dtype=float
    )

    # ---------------------------------------------------------
    # Create figure.
    # ---------------------------------------------------------

    fig = plt.figure(
        figsize=(11, 8)
    )

    ax = fig.add_subplot(
        111,
        projection="3d"
    )

    # ---------------------------------------------------------
    # Draw cameras.
    # ---------------------------------------------------------

    for index, pose in enumerate(camera_poses):

        view_number = int(
            pose["view"]
        )

        center = centers[index]

        rotation = rotations.get(
            view_number,
            np.eye(3)
        )

        draw_camera(
            ax,
            center,
            rotation,
            f"View {view_number}"
        )

    # ---------------------------------------------------------
    # Connect cameras in acquisition order.
    # ---------------------------------------------------------

    ax.plot(
        centers[:, 0],
        centers[:, 1],
        centers[:, 2],
        linestyle="--",
        linewidth=1.5,
        label="Camera path"
    )

    # ---------------------------------------------------------
    # Reference planar object.
    #
    # We deliberately use relative SFM coordinates rather than
    # pretending these dimensions are millimeters.
    # ---------------------------------------------------------

    x_min = -0.42
    x_max = 0.42

    y_min = -0.42
    y_max = 0.42

    z_plane = 1.0

    plane_x = np.array(
        [
            x_min,
            x_max,
            x_max,
            x_min,
            x_min
        ]
    )

    plane_y = np.array(
        [
            y_min,
            y_min,
            y_max,
            y_max,
            y_min
        ]
    )

    plane_z = np.full(
        plane_x.shape,
        z_plane
    )

    ax.plot(
        plane_x,
        plane_y,
        plane_z,
        linewidth=2.0,
        label="Reference planar object"
    )

    # ---------------------------------------------------------
    # Title.
    # ---------------------------------------------------------

    ax.set_title(
        "Relative Structure-from-Motion Camera Configuration\n"
        "Four Views of a Planar A4 Object",
        fontsize=13,
        pad=18
    )

    # ---------------------------------------------------------
    # Axis labels.
    # ---------------------------------------------------------

    ax.set_xlabel(
        "X (relative SFM units)"
    )

    ax.set_ylabel(
        "Y (relative SFM units)"
    )

    ax.set_zlabel(
        "Z (relative SFM units)"
    )

    # ---------------------------------------------------------
    # Equal-ish axis scaling.
    # ---------------------------------------------------------

    all_points = np.vstack(
        [
            centers,
            np.column_stack(
                [
                    plane_x,
                    plane_y,
                    plane_z
                ]
            )
        ]
    )

    minimums = all_points.min(
        axis=0
    )

    maximums = all_points.max(
        axis=0
    )

    midpoint = (
        minimums + maximums
    ) / 2.0

    radius = max(
        maximums - minimums
    ) / 2.0

    radius = max(
        radius,
        0.5
    )

    ax.set_xlim(
        midpoint[0] - radius,
        midpoint[0] + radius
    )

    ax.set_ylim(
        midpoint[1] - radius,
        midpoint[1] + radius
    )

    ax.set_zlim(
        midpoint[2] - radius,
        midpoint[2] + radius
    )

    # ---------------------------------------------------------
    # Camera viewing angle.
    # ---------------------------------------------------------

    ax.view_init(
        elev=25,
        azim=-55
    )

    ax.legend(
        loc="upper left"
    )

    # ---------------------------------------------------------
    # Important scientific note.
    # ---------------------------------------------------------

    fig.text(
        0.5,
        0.015,
        "Camera positions are relative and scale-ambiguous; "
        "they are not absolute physical distances.",
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

    # ---------------------------------------------------------
    # Save.
    # ---------------------------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    fig.savefig(
        OUTPUT_FILE,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(fig)


def main():

    results = load_results()

    print(
        f"Loaded: {RESULTS_FILE}"
    )

    print(
        "Camera poses found:",
        len(results["camera_poses"])
    )

    print(
        "Decompositions found:",
        len(results["decomposition"])
    )

    make_plot(
        results
    )

    print()
    print(
        f"Saved: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()
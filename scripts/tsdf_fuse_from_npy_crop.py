#!/usr/bin/env python3

import numpy as np
import open3d as o3d
import glob
import os
import sys

# ==============================
# SESSION PATH
# ==============================

# SESSION_PATH = "/home/bozznyskrtt/pcl_ws/teddybear/session17/test_outputs_knn_filtered/cleaned_depth"
# DEPTH_PATTERN = "frame_*.png"
# POSE_FOLDER = "poses"

DEFAULT_SESSION_ROOT = "/home/bozznyskrtt/pcl_ws/teddybear/session17"
SESSION_ROOT = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_SESSION_ROOT
SESSION_PATH = SESSION_ROOT + "/test_outputs_knn_filtered/cleaned_depth"

DEPTH_PATTERN = "frame_*.png"
POSE_FOLDER = "poses"

OUTPUT_MESH = "tsdf_mesh.ply"
OUTPUT_POINTCLOUD = "tsdf_cloud.ply"

# ==============================
# CAMERA INTRINSICS
# ==============================

WIDTH = 640
HEIGHT = 480

FX = 456.701904296875
FY = 456.701904296875
CX = 331.47186279296875
CY = 242.0

# ==============================
# DEPTH SETTINGS
# ==============================

DEPTH_SCALE = 1000.0      # uint16 depth in mm
DEPTH_TRUNC = 0.8         # meters

# ==============================
# TSDF PARAMETERS
# ==============================

VOXEL_SIZE = 0.003       # 5 mm
SDF_TRUNC = 0.02          # 2 cm

# ==============================
# CROP SETTINGS
# image crop: x_min, y_min, x_max, y_max
# Keep only this box, zero everything else
# ==============================

USE_CROP = False
CROP_X_MIN = 140
CROP_Y_MIN = 120
CROP_X_MAX = 380
CROP_Y_MAX = 410

# Optional depth range crop in meters
USE_DEPTH_RANGE_CROP = True
CROP_DEPTH_MIN_M = 0.20
CROP_DEPTH_MAX_M = 0.80


def crop_depth_image(depth_np: np.ndarray) -> np.ndarray:
    out = depth_np.copy()

    if USE_CROP:
        mask = np.zeros_like(out, dtype=bool)
        mask[CROP_Y_MIN:CROP_Y_MAX, CROP_X_MIN:CROP_X_MAX] = True
        out[~mask] = 0

    if USE_DEPTH_RANGE_CROP:
        depth_m = out.astype(np.float32) / DEPTH_SCALE
        bad = (depth_m < CROP_DEPTH_MIN_M) | (depth_m > CROP_DEPTH_MAX_M)
        out[bad] = 0

    return out


def make_rgbd_from_depth(depth_np: np.ndarray) -> o3d.geometry.RGBDImage:
    depth_img = o3d.geometry.Image(depth_np.astype(np.uint16))
    dummy_color = o3d.geometry.Image(
        np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    )
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        dummy_color,
        depth_img,
        depth_scale=DEPTH_SCALE,
        depth_trunc=DEPTH_TRUNC,
        convert_rgb_to_intensity=False
    )
    return rgbd


def main():
    depth_files = sorted(glob.glob(os.path.join(SESSION_PATH, DEPTH_PATTERN)))
    if len(depth_files) == 0:
        print("No depth images found.")
        return

    poses_dir = os.path.join(SESSION_ROOT, POSE_FOLDER)

    intrinsics = o3d.camera.PinholeCameraIntrinsic(
        WIDTH, HEIGHT, FX, FY, CX, CY
    )

    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=VOXEL_SIZE,
        sdf_trunc=SDF_TRUNC,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.NoColor
    )

    integrated = 0

    for depth_path in depth_files:
        name = os.path.splitext(os.path.basename(depth_path))[0]
        pose_path = os.path.join(poses_dir, f"{name}_T_ee_to_cam.npy")

        if not os.path.exists(pose_path):
            print(f"[SKIP] Missing pose: {pose_path}")
            continue

        # Load pose
        extrinsic = np.linalg.inv(np.load(pose_path))

        # Load and crop depth
        depth_o3d = o3d.io.read_image(depth_path)
        depth_np = np.asarray(depth_o3d)

        if depth_np.dtype != np.uint16:
            depth_np = depth_np.astype(np.uint16)

        depth_np = crop_depth_image(depth_np)

        # Skip empty frames
        if np.count_nonzero(depth_np) < 100:
            print(f"[SKIP] Too few valid pixels after crop: {depth_path}")
            continue

        rgbd = make_rgbd_from_depth(depth_np)

        # Integrate
        volume.integrate(rgbd, intrinsics, extrinsic)
        integrated += 1

        if integrated % 10 == 0:
            print(f"Integrated {integrated} frames")

    print(f"Total integrated: {integrated}")

    if integrated == 0:
        print("No frames integrated.")
        return

    # ==============================
    # Extract mesh
    # ==============================
    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()

    mesh_path = os.path.join(SESSION_ROOT, OUTPUT_MESH)
    o3d.io.write_triangle_mesh(mesh_path, mesh)
    print(f"Mesh saved to: {mesh_path}")

    # ==============================
    # Extract point cloud
    # ==============================
    pcd = volume.extract_point_cloud()

    # # Optional cleanup
    # if len(pcd.points) > 0:
    #     pcd = pcd.voxel_down_sample(voxel_size=0.0005)

    pcd_path = os.path.join(SESSION_ROOT, OUTPUT_POINTCLOUD)
    o3d.io.write_point_cloud(pcd_path, pcd)
    print(f"Point cloud saved to: {pcd_path}")

    # Optional preview
    # o3d.visualization.draw_geometries([mesh])
    # o3d.visualization.draw_geometries([pcd])


if __name__ == "__main__":
    main()
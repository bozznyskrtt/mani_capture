#!/usr/bin/env python3

import sys
import numpy as np
import open3d as o3d
import glob
import os

# ==============================
# SESSION PATH
# ==============================

DEFAULT_SESSION_ROOT = "/home/bozznyskrtt/pcl_ws/teddybear/session17"
SESSION_ROOT = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_SESSION_ROOT
SESSION_PATH = SESSION_ROOT + "/test_outputs_knn_filtered/cleaned_depth"

DEPTH_PATTERN = "frame_*.png"
POSE_FOLDER = "poses"

OUTPUT_MESH = "tsdf_mesh.ply"

# ==============================
# CAMERA INTRINSICS
# from /camera/depth/camera_info
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

DEPTH_SCALE = 1000.0      # uint16 mm -> meters
DEPTH_TRUNC = 1.2         # ignore >1.2m

# ==============================
# TSDF PARAMETERS
# ==============================

VOXEL_SIZE = 0.003        # 3mm
SDF_TRUNC = 0.01          # 1cm


USE_CROP = True
CROP_X_MIN = 170
CROP_Y_MIN = 130
CROP_X_MAX = 440
CROP_Y_MAX = 360

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

    depth_files = sorted(
        glob.glob(os.path.join(SESSION_PATH, DEPTH_PATTERN))
    )

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

    dummy_color = o3d.geometry.Image(
        np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    )

    integrated = 0

    for depth_path in depth_files:

        name = os.path.splitext(os.path.basename(depth_path))[0]

        pose_path = os.path.join(
            poses_dir,
            f"{name}_T_ee_to_cam.npy"
        )

        if not os.path.exists(pose_path):
            print("Missing pose:", pose_path)
            continue

        extrinsic = np.linalg.inv(np.load(pose_path))

        # depth = o3d.io.read_image(depth_path)

        # Load and crop depth
        depth_o3d = o3d.io.read_image(depth_path)
        depth_np = np.asarray(depth_o3d)

        if depth_np.dtype != np.uint16:
            depth_np = depth_np.astype(np.uint16)

        depth_np = crop_depth_image(depth_np)

        # rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        #     dummy_color,
        #     depth,
        #     depth_scale=DEPTH_SCALE,
        #     depth_trunc=DEPTH_TRUNC,
        #     convert_rgb_to_intensity=False
        # )

        rgbd = make_rgbd_from_depth(depth_np)

        volume.integrate(
            rgbd,
            intrinsics,
            extrinsic
        )

        integrated += 1

        if integrated % 10 == 0:
            print("Integrated", integrated, "frames")

    print("Total integrated:", integrated)

    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()

    out_path = os.path.join(SESSION_PATH, OUTPUT_MESH)

    o3d.io.write_triangle_mesh(out_path, mesh)

    print("Mesh saved to:", out_path)


if __name__ == "__main__":
    main()
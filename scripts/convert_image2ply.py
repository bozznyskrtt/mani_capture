#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from pathlib import Path
import cv2
import numpy as np
import open3d as o3d


class ConvertImage2Ply(Node):
    def __init__(self):
        super().__init__('convert_image_ply')

        # ---- Parameters ----
        self.declare_parameter('input_dir', './')
        self.declare_parameter('output_dir', './ply_output')

        self.declare_parameter('fx', 456.701904296875)
        self.declare_parameter('fy', 456.701904296875)

        self.declare_parameter('cx', 331.47186279296875)
        self.declare_parameter('cy', 242.51722717285156)

        self.declare_parameter('depth_scale', 1000.0)
        self.declare_parameter('write_ascii', False)

        self._run_once_timer = self.create_timer(0.0, self._run_once)
        self._has_run = False

    def _run_once(self):
        if self._has_run:
            return
        self._has_run = True
        self._run_once_timer.cancel()

        input_dir = self.get_parameter('input_dir').value
        output_dir = self.get_parameter('output_dir').value
        fx = float(self.get_parameter('fx').value)
        fy = float(self.get_parameter('fy').value)
        cx = float(self.get_parameter('cx').value)
        cy = float(self.get_parameter('cy').value)
        depth_scale = float(self.get_parameter('depth_scale').value)
        write_ascii = bool(self.get_parameter('write_ascii').value)

        # ---- Validate params ----
        if fx <= 0.0 or fy <= 0.0:
            self.get_logger().error(
                f"fx/fy must be positive values: fx={fx}, fy={fy}"
            )
            return

        if depth_scale <= 0.0:
            self.get_logger().error(
                f"depth_scale must be a positive value: {depth_scale}"
            )
            return

        try:
            self.batch_convert_depth_to_ply(
                input_folder=input_dir,
                output_folder=output_dir,
                fx=fx,
                fy=fy,
                cx=cx,
                cy=cy,
                depth_scale=depth_scale,
                write_ascii=write_ascii,
            )
        except Exception as e:
            self.get_logger().exception(
                f"Exception occurred during conversion: {e}"
            )

    def batch_convert_depth_to_ply(
        self,
        input_folder: str,
        output_folder: str,
        fx: float,
        fy: float,
        cx: float = -1.0,
        cy: float = -1.0,
        depth_scale: float = 1000.0,
        write_ascii: bool = False,
    ):
        input_dir = Path(input_folder)
        output_dir = Path(output_folder)

        if not input_dir.exists() or not input_dir.is_dir():
            self.get_logger().error(
                f"Input directory does not exist or is invalid: {input_dir}"
            )
            return

        output_dir.mkdir(parents=True, exist_ok=True)

        # List PNG files
        png_files = sorted(input_dir.glob("*.png"))
        total_files = len(png_files)

        if total_files == 0:
            self.get_logger().warning(
                f"No PNG files found in [{input_folder}]."
            )
            return

        self.get_logger().info(f"Files to convert: {total_files}")

        # Read the first image to determine base size
        first_img = cv2.imread(str(png_files[0]), cv2.IMREAD_UNCHANGED)
        if first_img is None:
            self.get_logger().error(f"Failed to read image: {png_files[0]}")
            return

        if first_img.ndim != 2:
            self.get_logger().error(
                f"The first image is not a single-channel depth image: {png_files[0].name}"
            )
            return

        base_h, base_w = first_img.shape[:2]

        # If principal point is unspecified (negative), use image center
        if cx < 0.0:
            cx = (base_w - 1) / 2.0
        if cy < 0.0:
            cy = (base_h - 1) / 2.0

        self.get_logger().info(
            f"camera intrinsics: fx={fx}, fy={fy}, cx={cx}, cy={cy}, depth_scale={depth_scale}"
        )

        # Precompute pixel grid for the base size
        u_base, v_base = np.meshgrid(
            np.arange(base_w, dtype=np.float32),
            np.arange(base_h, dtype=np.float32)
        )

        converted = 0
        skipped = 0
        failed = 0

        for idx, file_path in enumerate(png_files, 1):
            depth_img = cv2.imread(str(file_path), cv2.IMREAD_UNCHANGED)
            if depth_img is None:
                self.get_logger().warning(
                    f"Skipped (read failed): {file_path.name}"
                )
                skipped += 1
                continue

            if depth_img.ndim != 2:
                self.get_logger().warning(
                    f"Skipped (not grayscale): {file_path.name}"
                )
                skipped += 1
                continue

            h, w = depth_img.shape[:2]

            # Handle image size mismatch
            if (h, w) != (base_h, base_w):
                self.get_logger().warning(
                    f"Skipped (image size mismatch): {file_path.name} "
                    f"[{w}x{h}] != base[{base_w}x{base_h}]"
                )
                skipped += 1
                continue

            # Valid depth pixels
            valid = depth_img > 0
            if not np.any(valid):
                self.get_logger().warning(
                    f"Skipped (no valid depth values): {file_path.name}"
                )
                skipped += 1
                continue

            # Convert depth values to meters
            z = depth_img[valid].astype(np.float32) / depth_scale

            # Back-project using pinhole camera model
            x = (u_base[valid] - cx) * z / fx
            y = (v_base[valid] - cy) * z / fy

            points = np.stack((x, y, z), axis=-1).astype(np.float32)

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)

            output_path = output_dir / f"{file_path.stem}.ply"

            success = o3d.io.write_point_cloud(
                str(output_path),
                pcd,
                write_ascii=write_ascii
            )

            if not success:
                self.get_logger().error(
                    f"Failed to write PLY: {output_path}"
                )
                failed += 1
                continue

            converted += 1

            if idx % 10 == 0 or idx == total_files:
                self.get_logger().info(
                    f"Progress: {idx}/{total_files} "
                    f"(converted={converted}, skipped={skipped}, failed={failed}) "
                    f"{file_path.name} -> {output_path.name}"
                )

        self.get_logger().info(
            "Conversion finished: "
            f"total={total_files}, converted={converted}, skipped={skipped}, failed={failed}"
        )


def main(args=None):
    rclpy.init(args=args)
    node = ConvertImage2Ply()

    # Spin to run the one-shot timer callback
    rclpy.spin_once(node, timeout_sec=0.1)
    # Extra spin for safety
    rclpy.spin_once(node, timeout_sec=0.1)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
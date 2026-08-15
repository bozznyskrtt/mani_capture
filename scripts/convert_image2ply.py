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

        self.declare_parameter(
            'input_dir',
            './depth_images'
        )
        self.declare_parameter(
            'output_dir',
            './ply_outputs'
        )

        self.declare_parameter(
            'fx',
            525.0
        )
        self.declare_parameter(
            'fy',
            525.0
        )
        self.declare_parameter(
            'cx',
            None
        )
        self.declare_parameter(
            'cy',
            None
        )
        self.declare_parameter(
            'depth_scale',
            1000.0
        )

        input_dir = self.get_parameter(
            'input_dir'
        ).value

        output_dir = self.get_parameter(
            'output_dir'
        ).value

        fx = self.get_parameter(
            'fx'
        ).value

        fy = self.get_parameter(
            'fy'
        ).value

        cx = self.get_parameter(
            'cx'
        ).value

        cy = self.get_parameter(
            'cy'
        ).value

        depth_scale = self.get_parameter(
            'depth_scale'
        ).value

        self.batch_convert_depth_to_ply(
            input_folder=input_dir,
            output_folder=output_dir,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            depth_scale=depth_scale,
        )

    def batch_convert_depth_to_ply(
        self,
        input_folder: str,
        output_folder: str,
        fx: float,
        fy: float,
        cx: float = None,
        cy: float = None,
        depth_scale: float = 1000.0,
    ):
        input_dir = Path(input_folder)
        output_dir = Path(output_folder)

        output_dir.mkdir(parents=True, exist_ok=True)

        # PNGファイル一覧
        png_files = sorted(input_dir.glob("*.png"))
        total_files = len(png_files)

        if total_files == 0:
            self.get_logger().warning(
                f"[{input_folder}] にPNGファイルが見つかりませんでした。"
            )
            return

        self.get_logger().info(
            f"変換対象: {total_files} 件"
        )

        # 1枚目から画像サイズを取得
        first_img = cv2.imread(
            str(png_files[0]),
            cv2.IMREAD_UNCHANGED
        )

        if first_img is None:
            self.get_logger().error(
                f"画像を読み込めません: {png_files[0]}"
            )
            return

        height, width = first_img.shape[:2]

        # 主点が指定されていなければ画像中央
        if cx is None:
            cx = width / 2.0

        if cy is None:
            cy = height / 2.0

        # ピクセル座標を事前計算
        u, v = np.meshgrid(
            np.arange(width),
            np.arange(height)
        )

        for idx, file_path in enumerate(png_files, 1):

            depth_img = cv2.imread(
                str(file_path),
                cv2.IMREAD_UNCHANGED
            )

            if depth_img is None:
                self.get_logger().warning(
                    f"スキップ（読み込み失敗）: {file_path.name}"
                )
                continue

            # 深度画像がカラーだった場合への対策
            if depth_img.ndim != 2:
                self.get_logger().warning(
                    f"スキップ（グレースケールではありません）: "
                    f"{file_path.name}"
                )
                continue

            # 有効な深度値
            valid = depth_img > 0

            if not np.any(valid):
                self.get_logger().warning(
                    f"スキップ（有効な深度値なし）: "
                    f"{file_path.name}"
                )
                continue

            # 深度値をメートルに変換
            z = depth_img[valid].astype(np.float32) / depth_scale

            # ピンホールカメラモデルによる逆投影
            x = (
                (u[valid] - cx)
                * z
                / fx
            )

            y = (
                (v[valid] - cy)
                * z
                / fy
            )

            points = np.stack(
                (x, y, z),
                axis=-1
            )

            # Open3D PointCloud
            pcd = o3d.geometry.PointCloud()

            pcd.points = o3d.utility.Vector3dVector(
                points
            )

            # PLY出力
            output_path = (
                output_dir / f"{file_path.stem}.ply"
            )

            success = o3d.io.write_point_cloud(
                str(output_path),
                pcd,
                write_ascii=False
            )

            if not success:
                self.get_logger().error(
                    f"PLY書き込み失敗: {output_path}"
                )
                continue

            if idx % 10 == 0 or idx == total_files:
                self.get_logger().info(
                    f"進捗: {idx}/{total_files} "
                    f"({file_path.name} -> "
                    f"{output_path.name})"
                )

        self.get_logger().info(
            "すべての変換が完了しました。"
        )


def main(args=None):

    rclpy.init(args=args)

    node = ConvertImage2Ply()

    # 今回は__init__内で処理が完了するため
    # spinは不要
    node.destroy_node()

    rclpy.shutdown()


if __name__ == "__main__":
    main()
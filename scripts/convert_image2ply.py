from pathlib import Path
import cv2
import numpy as np
import open3d as o3d


def batch_convert_depth_to_ply(
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

    # フォルダ内のPNGファイル一覧を取得
    png_files = sorted(list(input_dir.glob("*.png")))
    total_files = len(png_files)

    if total_files == 0:
        print(f"[{input_folder}] にPNGファイルが見つかりませんでした。")
        return

    print(f"変換対象: {total_files} 件")

    # 1枚目を読み込んでグリッド座標を事前計算（高速化）
    first_img = cv2.imread(str(png_files[0]), cv2.IMREAD_UNCHANGED)
    height, width = first_img.shape[:2]

    if cx is None:
        cx = width / 2.0
    if cy is None:
        cy = height / 2.0

    u, v = np.meshgrid(np.arange(width), np.arange(height))

    for idx, file_path in enumerate(png_files, 1):
        depth_img = cv2.imread(str(file_path), cv2.IMREAD_UNCHANGED)
        if depth_img is None:
            print(f"スキップ (読み込み失敗): {file_path.name}")
            continue

        # 有効な深度値（0より大きい値）のマスク
        valid = depth_img > 0
        if not np.any(valid):
            print(f"スキップ (有効な深度値なし): {file_path.name}")
            continue

        # 3D座標へ逆投影
        z = depth_img[valid] / depth_scale
        x = (u[valid] - cx) * z / fx
        y = (v[valid] - cy) * z / fy

        points = np.stack((x, y, z), axis=-1)

        # PLY書き出し (write_ascii=False でバイナリ保存し高速・省容量化)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        output_path = output_dir / f"{file_path.stem}.ply"
        o3d.io.write_point_cloud(str(output_path), pcd, write_ascii=False)

        if idx % 10 == 0 or idx == total_files:
            print(f"進捗: {idx}/{total_files} ({file_path.name} -> {output_path.name})")

    print("\nすべての変換が完了しました。")


# --- 実行設定 ---
if __name__ == "__main__":
    INPUT_DIR = "./depth_images"  # 深度PNGが入っているフォルダ
    OUTPUT_DIR = "./ply_outputs"  # PLYの出力先フォルダ

    # カメラパラメータ（お使いの環境に合わせて変更）
    FX = 525.0
    FY = 525.0
    DEPTH_SCALE = 1000.0  # 1mm単位の場合は1000.0

    batch_convert_depth_to_ply(
        input_folder=INPUT_DIR,
        output_folder=OUTPUT_DIR,
        fx=FX,
        fy=FY,
        depth_scale=DEPTH_SCALE,
    )

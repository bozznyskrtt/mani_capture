#!/usr/bin/env python3

import os
import cv2
import numpy as np
import argparse


def crop_depth_by_range(depth_img, min_depth, max_depth):
    """
    Keep pixels within depth range.
    Others set to 0.
    """
    mask = (depth_img >= min_depth) & (depth_img <= max_depth)

    cropped = np.zeros_like(depth_img)
    cropped[mask] = depth_img[mask]

    return cropped


def process_folder(indir, outdir, min_depth, max_depth):

    os.makedirs(outdir, exist_ok=True)

    files = sorted([f for f in os.listdir(indir) if f.endswith(".png")])

    for f in files:

        path = os.path.join(indir, f)
        depth = cv2.imread(path, cv2.IMREAD_UNCHANGED)

        if depth is None:
            print("skip", f)
            continue

        cropped = crop_depth_by_range(depth, min_depth, max_depth)

        out_path = os.path.join(outdir, f)
        cv2.imwrite(out_path, cropped)

        print("saved:", out_path)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--indir", required=True, help="input folder of depth images")
    parser.add_argument("--outdir", required=True, help="output folder")
    parser.add_argument("--min_depth", type=int, default=0)
    parser.add_argument("--max_depth", type=int, default=2000)

    args = parser.parse_args()

    process_folder(
        args.indir,
        args.outdir,
        args.min_depth,
        args.max_depth
    )
'''
    python3 depth_crop.py \
        --indir /home/bozznyskrtt/pcl_ws/teddybear/session1 \
        --outdir /home/bozznyskrtt/pcl_ws/teddybear/session1_crop \
        --min_depth 350  \
        --max_depth 500
'''
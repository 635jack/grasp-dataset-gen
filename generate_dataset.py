#!/usr/bin/env python3
"""
Grasp Dataset Generator — Main entry point.

Usage:
    python generate_dataset.py [--glb_dir data/glb] [--output_dir output] [--seed 42]

Place GLB files in data/glb/ and run this script to generate:
  - Monocular RGB renders
  - Grasp contact points with normals, tangents, and finger labels
"""
import argparse
import os
import sys

# Ensure pyrender uses offscreen rendering
# On macOS, pyrender needs a windowed context — don't force EGL.
# On Linux servers, set PYOPENGL_PLATFORM=egl or osmesa before running.
if sys.platform == "linux":
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

from grasp_dataset_gen.config import (
    DatasetConfig, CameraConfig, GraspConfig, GraspStrategy
)
from grasp_dataset_gen.dataset import generate_dataset


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate grasp dataset from GLB 3D objects",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python generate_dataset.py
  python generate_dataset.py --glb_dir ./my_objects --output_dir ./my_dataset
  python generate_dataset.py --strategies front_back left_right
  python generate_dataset.py --resolution 1024 768 --fov 60
        """,
    )
    parser.add_argument("--glb_dir", default="data/glb",
                        help="Directory containing .glb files (default: data/glb)")
    parser.add_argument("--output_dir", default="output",
                        help="Output directory (default: output)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--resolution", type=int, nargs=2, default=[640, 480],
                        metavar=("W", "H"),
                        help="Image resolution (default: 640 480)")
    parser.add_argument("--fov", type=float, default=45.0,
                        help="Camera field of view in degrees (default: 45)")
    parser.add_argument("--strategies", nargs="+",
                        default=["front_back", "left_right", "right_left"],
                        choices=["front_back", "left_right", "right_left"],
                        help="Grasp strategies to generate")
    parser.add_argument("--radius_factor", type=float, default=1.2,
                        help="Virtual cylinder radius factor vs bounding sphere (default: 1.2)")
    return parser.parse_args()


def main():
    args = parse_args()

    cam = CameraConfig(
        width=args.resolution[0],
        height=args.resolution[1],
        fov=args.fov,
    )

    strategies = [GraspStrategy(s) for s in args.strategies]
    grasp = GraspConfig(
        strategies=strategies,
        grasp_radius_factor=args.radius_factor,
    )

    config = DatasetConfig(
        camera=cam,
        grasp=grasp,
        glb_dir=args.glb_dir,
        output_dir=args.output_dir,
        seed=args.seed,
    )

    generate_dataset(config)


if __name__ == "__main__":
    main()

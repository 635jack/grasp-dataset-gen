"""
Configuration for Grasp Dataset Generation.

Defines camera placement, grasp strategies, image resolution,
and finger layout parameters.
"""
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Dict
from enum import Enum


class GraspStrategy(Enum):
    """
    Grasp approach direction relative to the camera viewpoint.

    FRONT_BACK:  thumb on the camera-facing (front) side,
                 4 fingers on the occluded (back) side.
    LEFT_RIGHT:  thumb on the left, 4 fingers on the right,
                 palm facing the camera.
    RIGHT_LEFT:  thumb on the right, 4 fingers on the left,
                 palm facing the camera.
    """
    FRONT_BACK = "front_back"
    LEFT_RIGHT = "left_right"
    RIGHT_LEFT = "right_left"


# Finger labels used in the dataset
FINGER_LABELS = ["thumb", "index", "middle", "ring", "pinky", "palm"]


@dataclass
class CameraConfig:
    """Camera setup for monocular rendering."""
    # Camera position in world space (looking at origin)
    position: Tuple[float, float, float] = (0.0, -0.4, 0.15)
    # Look-at target
    target: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    # Up direction
    up: Tuple[float, float, float] = (0.0, 0.0, 1.0)
    # Field of view in degrees
    fov: float = 45.0
    # Image resolution
    width: int = 640
    height: int = 480
    # Near/far clipping planes
    znear: float = 0.01
    zfar: float = 10.0


@dataclass
class GraspConfig:
    """Configuration for grasp point generation."""
    # Number of contact points per finger (set to 1 for precise keypoints)
    points_per_finger: int = 1
    # Number of palm contact points
    points_palm: int = 1
    # Grasp strategies to generate
    strategies: List[GraspStrategy] = field(default_factory=lambda: [
        GraspStrategy.FRONT_BACK,
        GraspStrategy.LEFT_RIGHT,
        GraspStrategy.RIGHT_LEFT,
    ])
    # Approximate grasp radius factor (relative to object bounding sphere)
    grasp_radius_factor: float = 1.2
    # Angular spread of fingers (radians)
    finger_spread: float = np.pi / 3  # 60 degrees total spread for 4 fingers
    # Noise on contact ray directions (set to 0.0 for deterministic keypoints)
    ray_noise_std: float = 0.0


@dataclass
class DatasetConfig:
    """Top-level configuration."""
    camera: CameraConfig = field(default_factory=CameraConfig)
    grasp: GraspConfig = field(default_factory=GraspConfig)
    # Input directory for GLB files
    glb_dir: str = "data/glb"
    # Output directory
    output_dir: str = "output"
    # Random seed
    seed: int = 42
    # Background color for rendering (RGBA)
    bg_color: Tuple[float, float, float, float] = (0.85, 0.85, 0.85, 1.0)
    # Lighting intensity
    light_intensity: float = 3.0

"""
Grasp point sampler: generate contact points on a mesh surface
that simulate a hand grasping the object.

For each grasp strategy (front/back, left/right), we:
1. Define finger approach directions in camera coordinates
2. Cast rays from outside the object toward its center
3. Find ray-mesh intersections = contact points
4. Compute surface normal and tangent at each contact
5. Label each point with its finger identity

Output format per contact point:
  - position (x, y, z) in world frame
  - normal (nx, ny, nz) unit outward surface normal
  - tangent (tx, ty, tz) unit vector in tangent plane (slip direction)
  - finger label string
"""
import numpy as np
import trimesh
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

from .config import (
    GraspConfig, GraspStrategy, CameraConfig,
    FINGER_LABELS
)
from .renderer import build_camera_pose


@dataclass
class ContactPoint:
    """A single grasp contact point on the mesh surface."""
    position: np.ndarray       # (3,) world XYZ
    normal: np.ndarray         # (3,) outward unit normal
    tangent: np.ndarray        # (3,) unit tangent in surface plane
    finger: str                # one of FINGER_LABELS

    def to_dict(self) -> dict:
        return {
            "position": self.position.tolist(),
            "normal": self.normal.tolist(),
            "tangent": self.tangent.tolist(),
            "finger": self.finger,
        }


def _compute_tangent(normal: np.ndarray, slip_hint: np.ndarray) -> np.ndarray:
    """
    Compute a unit tangent vector lying in the surface plane
    defined by `normal`, oriented as close to `slip_hint` as possible.

    Gram-Schmidt: t = slip_hint - (slip_hint . n) * n, then normalize.
    """
    t = slip_hint - np.dot(slip_hint, normal) * normal
    norm = np.linalg.norm(t)
    if norm < 1e-8:
        # Fallback: pick an arbitrary perpendicular
        arbitrary = np.array([1, 0, 0]) if abs(normal[0]) < 0.9 else np.array([0, 1, 0])
        t = np.cross(normal, arbitrary)
        norm = np.linalg.norm(t)
    return t / norm


def _directions_for_strategy(
    strategy: GraspStrategy,
    cam_config: CameraConfig,
) -> Dict[str, np.ndarray]:
    """
    Compute approach direction (unit vector, world frame) for each finger
    given a grasp strategy and the camera setup.

    Returns dict mapping finger label -> approach direction (toward center).
    The thumb is always on the opposite side from the 4 fingers.
    """
    # Build camera frame vectors
    cam_pose = build_camera_pose(cam_config.position, cam_config.target, cam_config.up)
    cam_right = cam_pose[:3, 0]   # X of camera = right in image
    cam_up = cam_pose[:3, 1]      # Y of camera = up in image
    cam_forward = -cam_pose[:3, 2]  # camera looks along -Z => forward

    if strategy == GraspStrategy.FRONT_BACK:
        # Thumb approaches from front (camera side), fingers from back
        thumb_dir = cam_forward           # toward object from the front
        fingers_dir = -cam_forward        # toward object from the back
        palm_dir = cam_forward * 0.3 + cam_up * 0.1  # slightly from front-bottom
        slip_hint = cam_up               # slip direction = vertical

    elif strategy == GraspStrategy.LEFT_RIGHT:
        # Thumb on left, fingers on right, palm facing camera
        thumb_dir = cam_right             # from left side
        fingers_dir = -cam_right          # from right side
        palm_dir = cam_forward            # palm faces camera
        slip_hint = cam_up

    elif strategy == GraspStrategy.RIGHT_LEFT:
        # Thumb on right, fingers on left, palm facing camera
        thumb_dir = -cam_right
        fingers_dir = cam_right
        palm_dir = cam_forward
        slip_hint = cam_up

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    # Normalize
    for v in [thumb_dir, fingers_dir, palm_dir, slip_hint]:
        v /= np.linalg.norm(v)

    return {
        "thumb": thumb_dir,
        "index": fingers_dir,
        "middle": fingers_dir,
        "ring": fingers_dir,
        "pinky": fingers_dir,
        "palm": palm_dir,
        "slip_hint": slip_hint,
    }


def _spread_finger_directions(
    base_dir: np.ndarray,
    up: np.ndarray,
    n_fingers: int,
    total_spread: float,
) -> List[np.ndarray]:
    """
    Spread n_fingers evenly around `base_dir` in the plane defined
    by base_dir and up. Returns list of unit direction vectors.
    """
    # Create perpendicular axis for rotation
    perp = np.cross(base_dir, up)
    norm_perp = np.linalg.norm(perp)
    if norm_perp < 1e-8:
        arbitrary = np.array([1, 0, 0]) if abs(base_dir[0]) < 0.9 else np.array([0, 1, 0])
        perp = np.cross(base_dir, arbitrary)
    perp /= np.linalg.norm(perp)

    # Also spread slightly along up direction
    angles = np.linspace(-total_spread / 2, total_spread / 2, n_fingers)
    directions = []
    for angle in angles:
        # Rotate base_dir around perp axis by angle
        rot = trimesh.transformations.rotation_matrix(angle, perp)[:3, :3]
        d = rot @ base_dir
        d /= np.linalg.norm(d)
        directions.append(d)
    return directions


def _cast_rays_to_surface(
    mesh: trimesh.Trimesh,
    center: np.ndarray,
    approach_dir: np.ndarray,
    n_points: int,
    radius: float,
    rng: np.random.Generator,
    noise_std: float = 0.02,
) -> List[Tuple[np.ndarray, np.ndarray, int]]:
    """
    Cast rays from outside the mesh toward its center along `approach_dir`
    to find contact points on the surface.

    Parameters
    ----------
    mesh : trimesh.Trimesh
    center : (3,) object center
    approach_dir : (3,) unit approach direction (finger -> object)
    n_points : number of contact points to generate
    radius : distance from center to start ray origins
    rng : numpy random generator
    noise_std : std of angular noise on rays

    Returns
    -------
    List of (position, face_normal, face_index) tuples
    """
    # Ray origins: start from opposite side of approach direction
    origin_base = center - approach_dir * radius

    # Create slightly scattered origins around the base
    contacts = []
    attempts = 0
    max_attempts = n_points * 10

    while len(contacts) < n_points and attempts < max_attempts:
        attempts += 1
        # Random perturbation perpendicular to approach_dir
        noise = rng.normal(0, radius * 0.15, size=3)
        noise -= np.dot(noise, approach_dir) * approach_dir  # project to perp plane
        origin = origin_base + noise

        # Add small angular noise to direction
        dir_noise = rng.normal(0, noise_std, size=3)
        direction = approach_dir + dir_noise
        direction /= np.linalg.norm(direction)

        # Ray-mesh intersection
        locations, index_ray, index_tri = mesh.ray.intersects_location(
            [origin], [direction]
        )

        if len(locations) > 0:
            # Take first intersection (closest to ray origin)
            dists = np.linalg.norm(locations - origin, axis=1)
            closest = np.argmin(dists)
            pt = locations[closest]
            tri_idx = index_tri[closest]
            face_normal = mesh.face_normals[tri_idx]

            # Ensure normal points outward (away from center)
            if np.dot(face_normal, pt - center) < 0:
                face_normal = -face_normal

            contacts.append((pt, face_normal, tri_idx))

    return contacts


class GraspSampler:
    """
    Samples grasp contact points on a mesh surface.

    For each grasp strategy, generates contact points for all 6 zones
    (thumb, index, middle, ring, pinky, palm) with surface normals,
    tangents, and finger labels.

    Parameters
    ----------
    config : GraspConfig
    cam_config : CameraConfig
    seed : int
    """

    def __init__(self, config: GraspConfig, cam_config: CameraConfig, seed: int = 42):
        self.config = config
        self.cam_config = cam_config
        self.rng = np.random.default_rng(seed)

    def sample(self, mesh: trimesh.Trimesh,
               strategy: GraspStrategy) -> List[ContactPoint]:
        """
        Generate grasp contact points for a given strategy.

        Parameters
        ----------
        mesh : trimesh.Trimesh
        strategy : GraspStrategy

        Returns
        -------
        contacts : list of ContactPoint
        """
        center = mesh.centroid
        # Bounding sphere radius
        bsphere_radius = np.max(np.linalg.norm(mesh.vertices - center, axis=1))
        ray_radius = bsphere_radius * self.config.grasp_radius_factor

        dirs = _directions_for_strategy(strategy, self.cam_config)
        slip_hint = dirs.pop("slip_hint")

        cam_pose = build_camera_pose(
            self.cam_config.position, self.cam_config.target, self.cam_config.up
        )
        cam_up = cam_pose[:3, 1]

        contacts: List[ContactPoint] = []

        for finger_label in FINGER_LABELS:
            base_dir = dirs[finger_label]
            n_pts = (self.config.points_palm if finger_label == "palm"
                     else self.config.points_per_finger)

            # Spread directions for the 4 back-fingers
            if finger_label in ("index", "middle", "ring", "pinky"):
                # Map each finger to its angular slot
                finger_order = ["index", "middle", "ring", "pinky"]
                idx = finger_order.index(finger_label)
                angle_offsets = np.linspace(
                    -self.config.finger_spread / 2,
                    self.config.finger_spread / 2,
                    4
                )
                offset = angle_offsets[idx]
                # Rotation axis = cam_up for horizontal spread
                rot_mat = trimesh.transformations.rotation_matrix(
                    offset, cam_up
                )[:3, :3]
                approach_dir = rot_mat @ base_dir
                approach_dir /= np.linalg.norm(approach_dir)
            else:
                approach_dir = base_dir.copy()

            # Cast rays
            raw_contacts = _cast_rays_to_surface(
                mesh, center, approach_dir,
                n_pts, ray_radius, self.rng,
                noise_std=self.config.ray_noise_std,
            )

            for pos, normal, _ in raw_contacts:
                tangent = _compute_tangent(normal, slip_hint)
                contacts.append(ContactPoint(
                    position=pos.copy(),
                    normal=normal.copy(),
                    tangent=tangent.copy(),
                    finger=finger_label,
                ))

        return contacts

"""
Grasp point sampler using a virtual cylinder approach with PCA-based axis detection.

Principle:
----------
For each mesh, we compute its **principal axis** via PCA on the vertices.
This axis is used as the cylinder axis around which fingers wrap.
The thumb direction is the camera-forward direction projected onto the plane
perpendicular to the principal axis (so contacts are always on the lateral
surface, not the end caps).

Anatomical angles from thumb direction (rotation around principal axis):

    thumb  :   0°   (front / thumb side)
    index  : 115°   (side, wrapping around)
    middle : 143°   (back-side)
    ring   : 168°   (mostly back)
    pinky  : 195°   (back, slight wrap to other side)
    palm   : 155°   (center of finger cluster, offset downward)
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


# ───────────────────────────────────────────────────────────
# Data container
# ───────────────────────────────────────────────────────────

@dataclass
class ContactPoint:
    """A single grasp contact point on the mesh surface."""
    position: np.ndarray       # (3,) world XYZ
    normal: np.ndarray         # (3,) outward unit normal
    tangent: np.ndarray        # (3,) unit tangent in surface plane
    finger: str                # one of FINGER_LABELS
    visibility: str = "UNKNOWN" # VISIBLE, OCCLUDED_BACK, OCCLUDED_FRONT, SILHOUETTE

    def to_dict(self) -> dict:
        return {
            "position": self.position.tolist(),
            "normal": self.normal.tolist(),
            "tangent": self.tangent.tolist(),
            "finger": self.finger,
            "visibility": self.visibility,
        }


# ───────────────────────────────────────────────────────────
# Anatomical angles
# ───────────────────────────────────────────────────────────

FINGER_ANGLES_DEG = {
    "thumb":  0.0,
    "index":  115.0,
    "middle": 143.0,
    "ring":   168.0,
    "pinky":  195.0,
    "palm":   155.0,   # angular position — also offset vertically in sample()
}


# ───────────────────────────────────────────────────────────
# Geometry helpers
# ───────────────────────────────────────────────────────────

def _compute_tangent(normal: np.ndarray, slip_hint: np.ndarray) -> np.ndarray:
    """Unit tangent in the surface plane, closest to slip_hint (Gram-Schmidt)."""
    t = slip_hint - np.dot(slip_hint, normal) * normal
    norm = np.linalg.norm(t)
    if norm < 1e-8:
        arbitrary = np.array([1, 0, 0]) if abs(normal[0]) < 0.9 else np.array([0, 1, 0])
        t = np.cross(normal, arbitrary)
        norm = np.linalg.norm(t)
    return t / norm


def _project_perp(v: np.ndarray, axis: np.ndarray) -> Optional[np.ndarray]:
    """
    Project v onto the plane perpendicular to axis and normalize.
    Returns None if the projected vector is degenerate.
    """
    p = v - np.dot(v, axis) * axis
    n = np.linalg.norm(p)
    return p / n if n > 1e-6 else None


def _mesh_principal_axis(mesh: trimesh.Trimesh) -> np.ndarray:
    """
    Find the principal axis (longest dimension) of the mesh via PCA on vertices.

    The returned axis is oriented so that it has a non-negative component
    along the world Z axis (or positive Y if Z ≈ 0).  This gives a consistent
    'upward' direction regardless of mesh orientation.
    """
    verts = mesh.vertices - mesh.centroid
    cov = np.cov(verts.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    # Largest eigenvector = principal axis (elongation direction)
    axis = eigenvectors[:, np.argmax(eigenvalues)]
    axis = axis / np.linalg.norm(axis)

    # Orient toward positive Z (if cylinder is vertical) or positive Y
    if axis[2] < -0.1:
        axis = -axis
    elif abs(axis[2]) < 0.1 and axis[1] < 0:
        axis = -axis
    return axis


def _build_finger_dirs(
    strategy: GraspStrategy,
    cam_config: CameraConfig,
    grasp_axis: np.ndarray,
) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """
    Build a dict {finger_label -> unit approach direction} by rotating
    the thumb direction around the mesh's principal axis by each
    anatomical angle.

    The thumb direction is the camera-forward direction projected onto
    the plane perpendicular to grasp_axis, so rays always hit the lateral
    surface of the object (not an end cap).

    Also returns a slip_hint vector for tangent computation.
    """
    cam_pose    = build_camera_pose(cam_config.position, cam_config.target, cam_config.up)
    cam_right   = cam_pose[:3, 0]
    cam_forward = -cam_pose[:3, 2]

    if strategy == GraspStrategy.FRONT_BACK:
        raw = _project_perp(cam_forward, grasp_axis)
        thumb_dir = raw if raw is not None else _project_perp(cam_right, grasp_axis)

    elif strategy == GraspStrategy.LEFT_RIGHT:
        raw = _project_perp(-cam_right, grasp_axis)
        thumb_dir = raw if raw is not None else _project_perp(cam_forward, grasp_axis)

    elif strategy == GraspStrategy.RIGHT_LEFT:
        raw = _project_perp(cam_right, grasp_axis)
        thumb_dir = raw if raw is not None else _project_perp(cam_forward, grasp_axis)

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    if thumb_dir is None:
        raise RuntimeError("Cannot compute perpendicular thumb direction — degenerate geometry.")

    # Rotate thumb direction by each anatomical angle around grasp_axis
    dirs: Dict[str, np.ndarray] = {}
    for finger, angle_deg in FINGER_ANGLES_DEG.items():
        rot = trimesh.transformations.rotation_matrix(
            np.radians(angle_deg), grasp_axis
        )[:3, :3]
        d = rot @ thumb_dir
        dirs[finger] = d / np.linalg.norm(d)

    # Tangent slip direction = along the principal axis
    slip_hint = grasp_axis.copy()

    return dirs, slip_hint


def _cast_ray_to_surface(
    mesh: trimesh.Trimesh,
    ray_origin_offset: np.ndarray,   # start = center + this - approach_dir*R
    approach_dir: np.ndarray,
    radius: float,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Cast one ray toward the mesh centroid + ray_origin_offset.
    Returns (contact_position, outward_normal) or None.
    """
    center     = mesh.centroid + ray_origin_offset
    origin     = center - approach_dir * radius
    direction  = approach_dir.copy()

    locations, _, index_tri = mesh.ray.intersects_location(
        [origin], [direction]
    )

    if len(locations) == 0:
        return None

    dists   = np.linalg.norm(locations - origin, axis=1)
    closest = np.argmin(dists)
    pt      = locations[closest]
    normal  = mesh.face_normals[index_tri[closest]].copy()

    # Make sure normal points away from the local ray origin center
    if np.dot(normal, pt - center) < 0:
        normal = -normal

    return pt, normal


# ───────────────────────────────────────────────────────────
# Sampler class
# ───────────────────────────────────────────────────────────

class GraspSampler:
    """
    Samples grasp contact points on a mesh using a PCA-based cylinder model.

    For each strategy:
    1. Detect the object's principal axis via PCA
    2. Project thumb direction perpendicular to this axis
    3. Rotate thumb direction by anatomical angles to get all finger directions
    4. Cast one ray per finger to find the surface contact

    Palm is shifted downward along the principal axis to simulate it being
    below the fingertip cluster.
    """

    def __init__(self, config: GraspConfig, cam_config: CameraConfig, seed: int = 42):
        self.config     = config
        self.cam_config = cam_config

    def sample(self, mesh: trimesh.Trimesh,
               strategy: GraspStrategy) -> List[ContactPoint]:
        """
        Generate one contact point per finger zone.

        Returns
        -------
        contacts : list of ContactPoint (up to 6, one per label in FINGER_LABELS)
        """
        center = mesh.centroid
        bsphere_radius = np.max(np.linalg.norm(mesh.vertices - center, axis=1))
        ray_radius = bsphere_radius * self.config.grasp_radius_factor

        # Detect object orientation
        grasp_axis = _mesh_principal_axis(mesh)

        # Compute approach directions for each finger
        approach_dirs, slip_hint = _build_finger_dirs(strategy, self.cam_config, grasp_axis)

        # Half-length of the object along its principal axis (for palm offset)
        half_length = np.max(np.abs(
            np.dot(mesh.vertices - center, grasp_axis)
        ))

        contacts: List[ContactPoint] = []

        for finger_label in FINGER_LABELS:
            approach_dir = approach_dirs[finger_label]

            # Palm is positioned lower along the principal axis
            # to simulate the base of the hand below the fingertips
            origin_offset = np.zeros(3)
            if finger_label == "palm":
                origin_offset = -grasp_axis * half_length * 0.45

            result = _cast_ray_to_surface(
                mesh, origin_offset, approach_dir, ray_radius
            )

            if result is None:
                print(f"  ⚠️  No hit for {finger_label} ({strategy.value}), skipping.")
                continue

            pos, normal = result
            tangent = _compute_tangent(normal, slip_hint)

            contacts.append(ContactPoint(
                position=pos,
                normal=normal,
                tangent=tangent,
                finger=finger_label,
            ))

        return contacts

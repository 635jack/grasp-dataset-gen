"""
I/O utilities: load GLB meshes, save contact data, visualize.
"""
import os
import json
import numpy as np
import trimesh
from typing import List, Optional
from PIL import Image, ImageDraw

from .grasp_sampler import ContactPoint
from .config import CameraConfig
from .renderer import build_camera_pose


def load_glb(path: str) -> trimesh.Trimesh:
    """
    Load a GLB file and return a single watertight trimesh.
    
    If the file contains a Scene (multiple meshes), they are
    concatenated into a single mesh.
    """
    loaded = trimesh.load(path, force='mesh')
    if isinstance(loaded, trimesh.Scene):
        meshes = [g for g in loaded.geometry.values()
                  if isinstance(g, trimesh.Trimesh)]
        if not meshes:
            raise ValueError(f"No meshes found in {path}")
        loaded = trimesh.util.concatenate(meshes)
    return loaded


def normalize_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """
    Center the mesh at the origin and scale it to fit in a unit sphere.
    Returns a new mesh (does not modify in-place).
    """
    mesh = mesh.copy()
    mesh.vertices -= mesh.centroid
    scale = np.max(np.linalg.norm(mesh.vertices, axis=1))
    if scale > 1e-8:
        mesh.vertices /= scale
        # Scale to ~0.15m radius (reasonable graspable object size)
        mesh.vertices *= 0.08
    return mesh


def save_contacts_json(contacts: List[ContactPoint], path: str,
                       strategy: str, mesh_name: str):
    """Save contact points to a structured JSON file."""
    data = {
        "mesh": mesh_name,
        "strategy": strategy,
        "n_contacts": len(contacts),
        "contacts": [c.to_dict() for c in contacts],
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def save_contacts_npz(contacts: List[ContactPoint], path: str):
    """
    Save contact points to a compressed numpy archive.
    
    Arrays stored:
      positions : (N, 3) float32
      normals   : (N, 3) float32
      tangents  : (N, 3) float32
      fingers   : (N,) string array
    """
    positions = np.array([c.position for c in contacts], dtype=np.float32)
    normals = np.array([c.normal for c in contacts], dtype=np.float32)
    tangents = np.array([c.tangent for c in contacts], dtype=np.float32)
    fingers = np.array([c.finger for c in contacts])

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savez_compressed(path,
                        positions=positions,
                        normals=normals,
                        tangents=tangents,
                        fingers=fingers)


def project_to_image(points_3d: np.ndarray,
                     cam_config: CameraConfig) -> np.ndarray:
    """
    Project 3D world points to 2D image pixel coordinates.
    
    Parameters
    ----------
    points_3d : (N, 3) world coordinates
    cam_config : camera configuration
    
    Returns
    -------
    pixels : (N, 2) pixel coordinates (u, v) — may be out of bounds
    """
    # Build view matrix (inverse of camera pose)
    cam_pose = build_camera_pose(cam_config.position, cam_config.target, cam_config.up)
    view = np.linalg.inv(cam_pose)

    # Perspective projection
    fov_rad = np.radians(cam_config.fov)
    f = 1.0 / np.tan(fov_rad / 2.0)
    aspect = cam_config.width / cam_config.height

    # Transform to camera space
    pts_h = np.hstack([points_3d, np.ones((len(points_3d), 1))])
    pts_cam = (view @ pts_h.T).T[:, :3]  # (N, 3) in camera space

    # Project: x_ndc = f * x / (-z * aspect), y_ndc = f * y / (-z)
    z = -pts_cam[:, 2]  # flip Z for OpenGL
    z = np.clip(z, 1e-6, None)

    x_ndc = f * pts_cam[:, 0] / (z * aspect)
    y_ndc = f * pts_cam[:, 1] / z

    # NDC [-1, 1] -> pixel coords
    u = (x_ndc + 1) * 0.5 * cam_config.width
    v = (1 - y_ndc) * 0.5 * cam_config.height  # flip Y for image

    return np.stack([u, v], axis=1)


# Color palette for fingers
FINGER_COLORS = {
    "thumb":  (139, 69, 19),    # Brown (Marron)
    "index":  (255, 255, 0),    # Yellow (Jaune)
    "middle": (255, 165, 0),    # Orange
    "ring":   (255, 0, 0),      # Red (Rouge)
    "pinky":  (0, 255, 0),      # Green (Vert)
    "palm":   (0, 0, 0),        # Black (Noir)
}


def overlay_contacts_on_image(
    image: np.ndarray,
    contacts: List[ContactPoint],
    cam_config: CameraConfig,
    dot_radius: int = 4,
) -> Image.Image:
    """
    Draw contact points on the RGB image, color-coded by finger.
    
    Parameters
    ----------
    image : (H, W, 3) uint8 array
    contacts : list of ContactPoint
    cam_config : CameraConfig
    dot_radius : pixel radius for each point
    
    Returns
    -------
    PIL Image with overlaid contact points
    """
    img = Image.fromarray(image).copy()
    draw = ImageDraw.Draw(img)

    if not contacts:
        return img

    positions = np.array([c.position for c in contacts])
    pixels = project_to_image(positions, cam_config)

    for px, contact in zip(pixels, contacts):
        u, v = int(px[0]), int(px[1])
        color = FINGER_COLORS.get(contact.finger, (200, 200, 200))
        draw.ellipse(
            [u - dot_radius, v - dot_radius,
             u + dot_radius, v + dot_radius],
            fill=color,
            outline=(255, 255, 255),
        )

    return img


def export_scene_to_glb(mesh: trimesh.Trimesh, contacts: List[ContactPoint]) -> bytes:
    """
    Merge the mesh with colored spheres for contacts and return GLB as bytes.
    Useful for 3D preview in Streamlit/web using model-viewer.
    """
    scene = trimesh.Scene()
    
    # Original mesh with subtle transparency
    m = mesh.copy()
    m.visual.face_colors = [180, 180, 190, 180]
    scene.add_geometry(m)
    
    for contact in contacts:
        color = FINGER_COLORS.get(contact.finger, (200, 200, 200))
        # Smaller spheres for 3D view
        sphere = trimesh.creation.uv_sphere(radius=0.002, count=[10, 10])
        sphere.apply_translation(contact.position)
        sphere.visual.face_colors = list(color) + [255]
        scene.add_geometry(sphere)
        
        # Add normal vector as a tiny stick
        if np.linalg.norm(contact.normal) > 1e-6:
            n_start = contact.position
            n_end = n_start + np.array(contact.normal) * 0.015
            stick = trimesh.creation.cylinder(radius=0.0005, segment=[n_start, n_end])
            stick.visual.face_colors = [255, 255, 255, 200]
            scene.add_geometry(stick)

    return scene.export(file_type='glb')

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
    Load a GLB file and return a single trimesh for geometry operations.
    Uses force='mesh' for speed and stability. No color preservation.
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
    Center the mesh at the origin and scale it to fit ~0.08m radius.
    Returns a new mesh (does not modify in-place).
    """
    verts = mesh.vertices.copy()
    verts -= verts.mean(axis=0)
    scale = np.max(np.linalg.norm(verts, axis=1))
    if scale > 1e-8:
        verts = verts / scale * 0.08
    return trimesh.Trimesh(vertices=verts, faces=mesh.faces.copy(), process=False)


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
    """
    cam_pose = build_camera_pose(cam_config.position, cam_config.target, cam_config.up)
    view = np.linalg.inv(cam_pose)

    fov_rad = np.radians(cam_config.fov)
    f = 1.0 / np.tan(fov_rad / 2.0)
    aspect = cam_config.width / cam_config.height

    pts_h = np.hstack([points_3d, np.ones((len(points_3d), 1))])
    pts_cam = (view @ pts_h.T).T[:, :3]

    z = -pts_cam[:, 2]
    z = np.clip(z, 1e-6, None)

    x_ndc = f * pts_cam[:, 0] / (z * aspect)
    y_ndc = f * pts_cam[:, 1] / z

    u = (x_ndc + 1) * 0.5 * cam_config.width
    v = (1 - y_ndc) * 0.5 * cam_config.height

    return np.stack([u, v], axis=1)


# Color palette for fingers
FINGER_COLORS = {
    "thumb":  (139, 69, 19),    # Brown
    "index":  (255, 255, 0),    # Yellow
    "middle": (255, 165, 0),    # Orange
    "ring":   (255, 0, 0),      # Red
    "pinky":  (0, 255, 0),      # Green
    "palm":   (0, 0, 0),        # Black
}


def overlay_contacts_on_image(
    image: np.ndarray,
    contacts: List[ContactPoint],
    cam_config: CameraConfig,
    dot_radius: int = 4,
) -> Image.Image:
    """
    Draw contact points on the RGB image, color-coded by finger.
    Occluded points get a cross (X), silhouette points get a double outline.
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
        
        # Base circle
        draw.ellipse(
            [u - dot_radius, v - dot_radius,
             u + dot_radius, v + dot_radius],
            fill=color,
            outline=(255, 255, 255),
        )

        # Cross for occluded points
        if contact.visibility and "OCCLUDED" in contact.visibility:
            d = dot_radius - 1
            draw.line([u - d, v - d, u + d, v + d], fill=(255, 255, 255), width=1)
            draw.line([u - d, v + d, u + d, v - d], fill=(255, 255, 255), width=1)
        
        # Double outline for silhouette points
        elif contact.visibility == "SILHOUETTE":
            draw.ellipse(
                [u - dot_radius - 1, v - dot_radius - 1,
                 u + dot_radius + 1, v + dot_radius + 1],
                outline=(255, 255, 255),
            )

    return img


def export_scene_to_glb(mesh: trimesh.Trimesh, contacts: List[ContactPoint]) -> bytes:
    """
    Merge the mesh with colored spheres for contacts and return GLB as bytes.
    Useful for 3D preview in Streamlit/web using model-viewer.
    """
    scene = trimesh.Scene()
    
    m = mesh.copy()
    m.visual.face_colors = [180, 180, 190, 180]
    scene.add_geometry(m)
    
    for contact in contacts:
        color = FINGER_COLORS.get(contact.finger, (200, 200, 200))
        sphere = trimesh.creation.uv_sphere(radius=0.002, count=[10, 10])
        sphere.apply_translation(contact.position)
        sphere.visual.face_colors = list(color) + [255]
        scene.add_geometry(sphere)
        
        if np.linalg.norm(contact.normal) > 1e-6:
            n_start = contact.position
            n_end = n_start + np.array(contact.normal) * 0.015
            stick = trimesh.creation.cylinder(radius=0.0005, segment=[n_start, n_end])
            stick.visual.face_colors = [255, 255, 255, 200]
            scene.add_geometry(stick)

    return scene.export(file_type='glb')

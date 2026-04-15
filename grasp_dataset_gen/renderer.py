"""
Off-screen renderer for monocular RGB images of 3D objects.

Uses pyrender with EGL/OSMesa headless backend to produce
clean RGB renders of GLB meshes from a configurable camera.
"""
import os
import numpy as np
import trimesh
import pyrender
from PIL import Image
from typing import Optional, Tuple

from .config import CameraConfig


def build_camera_pose(position: Tuple[float, float, float],
                      target: Tuple[float, float, float],
                      up: Tuple[float, float, float]) -> np.ndarray:
    """
    Build a 4x4 camera-to-world pose matrix (OpenGL convention).
    
    Camera looks along -Z in its local frame (OpenGL standard).
    
    Parameters
    ----------
    position : (3,) camera position in world
    target   : (3,) look-at point in world
    up       : (3,) world-up direction
    
    Returns
    -------
    pose : (4, 4) camera-to-world matrix
    """
    pos = np.array(position, dtype=np.float64)
    tgt = np.array(target, dtype=np.float64)
    up_vec = np.array(up, dtype=np.float64)

    # Camera looks at -Z, so forward = target - position (then we negate for Z)
    forward = tgt - pos
    forward /= np.linalg.norm(forward)

    right = np.cross(forward, up_vec)
    right /= np.linalg.norm(right)

    true_up = np.cross(right, forward)

    pose = np.eye(4, dtype=np.float64)
    pose[:3, 0] = right
    pose[:3, 1] = true_up
    pose[:3, 2] = -forward  # OpenGL: camera looks along -Z
    pose[:3, 3] = pos

    return pose


class MeshRenderer:
    """
    Renders a trimesh.Trimesh object to an RGB image using pyrender.
    
    Parameters
    ----------
    cam_config : CameraConfig
        Camera parameters (position, resolution, FOV, etc.)
    bg_color : tuple of 4 floats
        RGBA background color [0..1]
    light_intensity : float
        Intensity of the lighting rig
    """

    def __init__(self, cam_config: CameraConfig,
                 bg_color: Tuple[float, float, float, float] = (0.85, 0.85, 0.85, 1.0),
                 light_intensity: float = 3.0):
        self.cam_config = cam_config
        self.bg_color = bg_color
        self.light_intensity = light_intensity
        # Create the off-screen renderer ONCE and reuse it.
        # Pyglet on macOS crashes if you repeatedly create/destroy GL contexts.
        self._renderer = pyrender.OffscreenRenderer(
            viewport_width=cam_config.width,
            viewport_height=cam_config.height,
        )

    def close(self):
        """Release the OpenGL context."""
        if self._renderer is not None:
            self._renderer.delete()
            self._renderer = None

    def __del__(self):
        self.close()

    def render(self, mesh: trimesh.Trimesh,
               output_path: Optional[str] = None) -> np.ndarray:
        """
        Render the mesh and return the RGB image as a numpy array.
        
        Parameters
        ----------
        mesh : trimesh.Trimesh
            The 3D mesh to render.
        output_path : str, optional
            If provided, save the image to this path.
            
        Returns
        -------
        color : np.ndarray of shape (H, W, 3), dtype uint8
        """
        scene = pyrender.Scene(
            bg_color=self.bg_color,
            ambient_light=[0.3, 0.3, 0.3]
        )

        # Add mesh
        # Give a uniform material if none exists
        material = pyrender.MetallicRoughnessMaterial(
            baseColorFactor=[0.6, 0.6, 0.7, 1.0],
            metallicFactor=0.1,
            roughnessFactor=0.7,
        )
        py_mesh = pyrender.Mesh.from_trimesh(mesh, material=material)
        scene.add(py_mesh)

        # Camera
        cam = pyrender.PerspectiveCamera(
            yfov=np.radians(self.cam_config.fov),
            znear=self.cam_config.znear,
            zfar=self.cam_config.zfar,
            aspectRatio=self.cam_config.width / self.cam_config.height
        )
        cam_pose = build_camera_pose(
            self.cam_config.position,
            self.cam_config.target,
            self.cam_config.up,
        )
        scene.add(cam, pose=cam_pose)

        # Lighting: 3-point setup
        # Key light (front-left-top)
        key_pose = build_camera_pose((-0.5, -0.5, 0.8), (0, 0, 0), (0, 0, 1))
        key_light = pyrender.DirectionalLight(
            color=[1.0, 0.97, 0.95],
            intensity=self.light_intensity
        )
        scene.add(key_light, pose=key_pose)

        # Fill light (front-right, dimmer)
        fill_pose = build_camera_pose((0.6, -0.3, 0.3), (0, 0, 0), (0, 0, 1))
        fill_light = pyrender.DirectionalLight(
            color=[0.95, 0.95, 1.0],
            intensity=self.light_intensity * 0.5
        )
        scene.add(fill_light, pose=fill_pose)

        # Rim light (back-top)
        rim_pose = build_camera_pose((0.0, 0.5, 0.6), (0, 0, 0), (0, 0, 1))
        rim_light = pyrender.DirectionalLight(
            color=[1.0, 1.0, 1.0],
            intensity=self.light_intensity * 0.4
        )
        scene.add(rim_light, pose=rim_pose)

        # Render using the persistent off-screen renderer
        color, _ = self._renderer.render(scene)

        if output_path is not None:
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            Image.fromarray(color).save(output_path)

        return color

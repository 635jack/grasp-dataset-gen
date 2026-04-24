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
    """
    pos = np.array(position, dtype=np.float64)
    tgt = np.array(target, dtype=np.float64)
    up_vec = np.array(up, dtype=np.float64)

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


def _add_lights(scene, light_intensity):
    """Add 3-point lighting to a pyrender scene."""
    for pose, color, mult in [
        ((-0.5, -0.5, 0.8), [1.0, 0.97, 0.95], 1.0),
        ((0.6, -0.3, 0.3),  [0.95, 0.95, 1.0], 0.5),
        ((0.0, 0.5, 0.6),   [1.0, 1.0, 1.0],   0.4),
    ]:
        lp = build_camera_pose(pose, (0, 0, 0), (0, 0, 1))
        scene.add(pyrender.DirectionalLight(color=color, intensity=light_intensity * mult), pose=lp)


def _add_camera(scene, cam_config):
    """Add camera to a pyrender scene. Returns the pose matrix."""
    cam = pyrender.PerspectiveCamera(
        yfov=np.radians(cam_config.fov),
        znear=cam_config.znear,
        zfar=cam_config.zfar,
        aspectRatio=cam_config.width / cam_config.height
    )
    cam_pose = build_camera_pose(cam_config.position, cam_config.target, cam_config.up)
    scene.add(cam, pose=cam_pose)
    return cam_pose


# Neutral fallback material for meshes without colors
_FALLBACK_MATERIAL = pyrender.MetallicRoughnessMaterial(
    baseColorFactor=[0.72, 0.72, 0.75, 1.0],
    metallicFactor=0.1,
    roughnessFactor=0.8,
)


class MeshRenderer:
    """
    Renders trimesh objects to RGB images using pyrender.
    """

    def __init__(self, cam_config: CameraConfig,
                 bg_color: Tuple[float, float, float, float] = (0.85, 0.85, 0.85, 1.0),
                 light_intensity: float = 3.0):
        self.cam_config = cam_config
        self.bg_color = bg_color
        self.light_intensity = light_intensity
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

    def _new_scene(self):
        return pyrender.Scene(bg_color=self.bg_color, ambient_light=[0.3, 0.3, 0.3])

    def _do_render(self, scene, output_path=None):
        """Render scene and optionally save image."""
        color, depth = self._renderer.render(scene)
        if output_path:
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            Image.fromarray(color).save(output_path)
        return color, depth

    def render(self, mesh: trimesh.Trimesh,
               output_path: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Render a single trimesh. Uses neutral gray material.
        This is fast and stable — used after normalize_mesh().
        """
        scene = self._new_scene()
        py_mesh = pyrender.Mesh.from_trimesh(mesh, material=_FALLBACK_MATERIAL, smooth=False)
        scene.add(py_mesh)
        _add_camera(scene, self.cam_config)
        _add_lights(scene, self.light_intensity)
        return self._do_render(scene, output_path)

    def render_colored(self, glb_path: str, target_scale: float = 0.08,
                       output_path: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Render a GLB file preserving its native materials/colors.

        Strategy:
        1. Load the raw scene (not force='mesh') to preserve per-submesh materials.
        2. Normalize all geometries consistently.
        3. For each sub-mesh, try pyrender native rendering. If the texture causes
           an OpenGL error (glGenTextures bug on macOS), retry with fallback material.
        4. Return (color_image, depth_map).
        """
        try:
            loaded = trimesh.load(glb_path)
        except Exception as e:
            print(f"    ⚠️  Cannot load {glb_path}: {e}")
            return self._render_geometry_fallback(glb_path, output_path)

        scene = self._new_scene()

        # Collect all geometries
        if isinstance(loaded, trimesh.Scene):
            geoms = [g for g in loaded.geometry.values() if isinstance(g, trimesh.Trimesh)]
        elif isinstance(loaded, trimesh.Trimesh):
            geoms = [loaded]
        else:
            return self._render_geometry_fallback(glb_path, output_path)

        if not geoms:
            return self._render_geometry_fallback(glb_path, output_path)

        # Compute global normalization from all vertices
        all_verts = np.vstack([g.vertices for g in geoms])
        centroid = all_verts.mean(axis=0)
        scale = np.max(np.linalg.norm(all_verts - centroid, axis=1))
        if scale < 1e-8:
            scale = 1.0

        for geom in geoms:
            g = geom.copy()
            g.vertices = (g.vertices - centroid) / scale * target_scale

            # Try to render with native material/texture
            added = False
            try:
                py_mesh = pyrender.Mesh.from_trimesh(g, smooth=False)
                scene.add(py_mesh)
                added = True
            except Exception:
                pass

            if not added:
                # Fallback: force neutral material (strips textures)
                try:
                    py_mesh = pyrender.Mesh.from_trimesh(g, material=_FALLBACK_MATERIAL, smooth=False)
                    scene.add(py_mesh)
                except Exception:
                    pass  # skip this sub-mesh entirely

        _add_camera(scene, self.cam_config)
        _add_lights(scene, self.light_intensity)

        try:
            return self._do_render(scene, output_path)
        except Exception:
            # Even the render itself crashed (OpenGL) — full fallback
            return self._render_geometry_fallback(glb_path, output_path)

    def _render_geometry_fallback(self, glb_path, output_path=None):
        """Last resort: load with force='mesh' and render gray."""
        mesh = trimesh.load(glb_path, force='mesh')
        if isinstance(mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate(
                [g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
            )
        # Normalize
        verts = mesh.vertices.copy()
        verts -= verts.mean(axis=0)
        s = np.max(np.linalg.norm(verts, axis=1))
        if s > 1e-8:
            verts = verts / s * 0.08
        mesh_n = trimesh.Trimesh(vertices=verts, faces=mesh.faces.copy(), process=False)
        return self.render(mesh_n, output_path)

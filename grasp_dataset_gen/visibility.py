import numpy as np
import trimesh
from typing import List, Tuple
from .config import CameraConfig
from .renderer import build_camera_pose

def classify_visibility(
    points_3d: np.ndarray, 
    normals: np.ndarray, 
    cam_config: CameraConfig, 
    depth_map: np.ndarray, 
    silhouette_threshold: float = 0.1, 
    depth_epsilon: float = 0.002
) -> List[str]:
    """
    Classify 3D points based on their visibility from a specific camera.
    
    Returns a list of statuses:
    - VISIBLE: Facing camera and not occluded.
    - OCCLUDED_BACK: Back-facing (angle > 90 deg).
    - OCCLUDED_FRONT: Front-facing but hidden by another part of the mesh.
    - SILHOUETTE: Grazing angle (near 90 deg).
    - OUT_OF_FRAME: Point projects outside the camera sensor.
    """
    if len(points_3d) == 0:
        return []

    # 1. Transform to camera space for depth comparison
    cam_pose = build_camera_pose(cam_config.position, cam_config.target, cam_config.up)
    view_matrix = np.linalg.inv(cam_pose)
    
    pts_h = np.hstack([points_3d, np.ones((len(points_3d), 1))])
    pts_cam = (view_matrix @ pts_h.T).T[:, :3]
    
    # In Pyrender, camera looks at -Z. Depth map contains positive distance (-pts_cam[:, 2]).
    depths_real = -pts_cam[:, 2]
    
    # 2. Geometric check (Normal vs View)
    cam_pos = np.array(cam_config.position)
    view_dirs = cam_pos - points_3d
    # Avoid division by zero
    norms = np.linalg.norm(view_dirs, axis=1, keepdims=True)
    view_dirs = np.divide(view_dirs, norms, where=norms > 1e-8)
    
    dots = np.sum(normals * view_dirs, axis=1)
    
    # 3. Project to pixels
    from .utils import project_to_image
    pixels = project_to_image(points_3d, cam_config)
    
    results = []
    H, W = depth_map.shape
    
    for i, (px, d_real, dot) in enumerate(zip(pixels, depths_real, dots)):
        u, v = int(round(px[0])), int(round(px[1]))
        
        if not (0 <= u < W and 0 <= v < H):
            results.append("OUT_OF_FRAME")
            continue
            
        # Check silhouette first
        if abs(dot) < silhouette_threshold:
            results.append("SILHOUETTE")
        elif dot < 0:
            results.append("OCCLUDED_BACK")
        else:
            # Check Image-space occlusion (Z-buffer)
            d_buf = depth_map[v, u]
            # d_buf == 0 means background (no mesh rendered at this pixel)
            if d_buf > 0 and d_real > d_buf + depth_epsilon:
                results.append("OCCLUDED_FRONT")
            else:
                results.append("VISIBLE")
                
    return results

def calculate_surface_visibility(mesh: trimesh.Trimesh, cam_config: CameraConfig, depth_map: np.ndarray) -> float:
    """
    Calculate the ratio of the mesh surface that is visible from the camera.
    Uses vertex sampling as a proxy for area.
    """
    if len(mesh.vertices) == 0:
        return 0.0
        
    # Subsample vertices if the mesh is too dense for quick analysis
    max_pts = 5000
    if len(mesh.vertices) > max_pts:
        indices = np.random.choice(len(mesh.vertices), max_pts, replace=False)
        pts = mesh.vertices[indices]
        nls = mesh.vertex_normals[indices]
    else:
        pts = mesh.vertices
        nls = mesh.vertex_normals
        
    vis = classify_visibility(pts, nls, cam_config, depth_map)
    
    # Visible + Silhouette are considered "known" surface
    visible_count = sum(1 for v in vis if v in ["VISIBLE", "SILHOUETTE"])
    return visible_count / len(pts)

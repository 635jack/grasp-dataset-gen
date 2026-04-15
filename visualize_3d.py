#!/usr/bin/env python3
"""
Visualize the generated dataset: load contact points and display them
in a 3D interactive plot.
"""
import argparse
import json
import os
import numpy as np
import trimesh

from grasp_dataset_gen.utils import load_glb, normalize_mesh


FINGER_COLORS_RGB = {
    "thumb":  [139, 69, 19],
    "index":  [255, 255, 0],
    "middle": [255, 165, 0],
    "ring":   [255, 0, 0],
    "pinky":  [0, 255, 0],
    "palm":   [0, 0, 0],
}


def visualize_grasp(mesh_path: str, contacts_json: str):
    """
    Open a 3D viewer showing the mesh with contact points and normals.
    """
    # Load mesh
    mesh = load_glb(mesh_path)
    mesh = normalize_mesh(mesh)
    mesh.visual.face_colors = [180, 180, 190, 180]  # semi-transparent gray

    # Load contacts
    with open(contacts_json) as f:
        data = json.load(f)

    scene = trimesh.Scene([mesh])

    for contact in data["contacts"]:
        pos = np.array(contact["position"])
        normal = np.array(contact["normal"])
        tangent = np.array(contact["tangent"])
        finger = contact["finger"]
        color = FINGER_COLORS_RGB.get(finger, [200, 200, 200])

        # Contact point as a small sphere
        sphere = trimesh.creation.uv_sphere(radius=0.002, count=[8, 8])
        sphere.apply_translation(pos)
        sphere.visual.face_colors = color + [255]
        scene.add_geometry(sphere)

        # Normal vector as a cylinder (line)
        normal_end = pos + normal * 0.015
        normal_line = trimesh.creation.cylinder(
            radius=0.0005,
            segment=[pos, normal_end],
        )
        normal_line.visual.face_colors = [255, 255, 255, 200]
        scene.add_geometry(normal_line)

        # Tangent vector as a cylinder (cyan)
        tangent_end = pos + tangent * 0.01
        tangent_line = trimesh.creation.cylinder(
            radius=0.0004,
            segment=[pos, tangent_end],
        )
        tangent_line.visual.face_colors = [0, 255, 255, 180]
        scene.add_geometry(tangent_line)

    scene.show()


def main():
    parser = argparse.ArgumentParser(
        description="3D visualization of grasp contact points"
    )
    parser.add_argument("--glb", required=True, help="Path to the GLB file")
    parser.add_argument("--contacts", required=True,
                        help="Path to the contacts JSON file")
    args = parser.parse_args()

    visualize_grasp(args.glb, args.contacts)


if __name__ == "__main__":
    main()

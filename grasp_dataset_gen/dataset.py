"""
Dataset generation orchestrator.

Ties together mesh loading, rendering, and grasp sampling
to produce the full dataset.
"""
import os
import json
import glob
from tqdm import tqdm
from typing import Optional

from .config import DatasetConfig, GraspStrategy
from .renderer import MeshRenderer
from .grasp_sampler import GraspSampler
from .visibility import classify_visibility, calculate_surface_visibility
from .utils import (
    load_glb, normalize_mesh,
    save_contacts_json, save_contacts_npz,
    overlay_contacts_on_image,
)


def generate_dataset(config: Optional[DatasetConfig] = None):
    """
    Main entry point: iterate over all GLB files, render images,
    sample grasp contacts, and save everything.

    Directory structure produced:
        output/
        ├── <mesh_name>/
        │   ├── rgb.png                          # monocular RGB render
        │   ├── grasp_front_back.json             # contact points (JSON)
        │   ├── grasp_front_back.npz              # contact points (NPZ)
        │   ├── grasp_front_back_overlay.png      # visualization
        │   ├── grasp_left_right.json
        │   ├── grasp_left_right.npz
        │   ├── grasp_left_right_overlay.png
        │   ├── grasp_right_left.json
        │   ├── grasp_right_left.npz
        │   ├── grasp_right_left_overlay.png
        │   └── metadata.json                    # mesh info
        └── dataset_index.json                   # global index
    """
    if config is None:
        config = DatasetConfig()

    # Discover GLB files
    glb_pattern = os.path.join(config.glb_dir, "*.glb")
    glb_files = sorted(glob.glob(glb_pattern))

    if not glb_files:
        print(f"⚠️  No .glb files found in {config.glb_dir}")
        print(f"   Place your GLB objects in '{config.glb_dir}/' and re-run.")
        return

    print(f"🔍 Found {len(glb_files)} GLB files in {config.glb_dir}")
    print(f"📂 Output directory: {config.output_dir}")
    print(f"🖐️  Grasp strategies: {[s.value for s in config.grasp.strategies]}")
    print()

    # Init renderer and sampler
    renderer = MeshRenderer(
        config.camera,
        bg_color=config.bg_color,
        light_intensity=config.light_intensity,
    )
    sampler = GraspSampler(config.grasp, config.camera, seed=config.seed)

    dataset_index = []

    for glb_path in tqdm(glb_files, desc="Processing objects"):
        mesh_name = os.path.splitext(os.path.basename(glb_path))[0]
        out_dir = os.path.join(config.output_dir, mesh_name)
        os.makedirs(out_dir, exist_ok=True)

        # 1. Load and normalize
        print(f"\n📦 Loading {mesh_name}...")
        mesh = load_glb(glb_path)
        mesh = normalize_mesh(mesh)

        # 2. Render RGB image and Depth map
        rgb_path = os.path.join(out_dir, "rgb.png")
        print(f"  📷 Rendering images...")
        color_image, depth_map = renderer.render(mesh, output_path=rgb_path)

        # 2b. Object surface visibility
        import numpy as np
        surface_vis = calculate_surface_visibility(mesh, config.camera, depth_map)

        # 3. Generate grasp contacts for each strategy
        entry = {
            "mesh": mesh_name,
            "rgb": rgb_path,
            "surface_visibility": float(surface_vis),
            "grasps": {},
            "n_vertices": len(mesh.vertices),
            "n_faces": len(mesh.faces),
            "bounding_box": mesh.bounds.tolist(),
        }

        for strategy in config.grasp.strategies:
            print(f"  🖐️  Sampling grasp: {strategy.value}...")
            contacts = sampler.sample(mesh, strategy)
            
            # Visibility Analysis
            pts = np.array([c.position for c in contacts])
            nls = np.array([c.normal for c in contacts])
            v_statuses = classify_visibility(pts, nls, config.camera, depth_map)
            for c, status in zip(contacts, v_statuses):
                c.visibility = status

            strategy_name = strategy.value

            # Save JSON
            json_path = os.path.join(out_dir, f"grasp_{strategy_name}.json")
            save_contacts_json(contacts, json_path, strategy_name, mesh_name)

            # Save NPZ
            npz_path = os.path.join(out_dir, f"grasp_{strategy_name}.npz")
            save_contacts_npz(contacts, npz_path)

            # Overlay visualization
            overlay_path = os.path.join(out_dir, f"grasp_{strategy_name}_overlay.png")
            overlay_img = overlay_contacts_on_image(
                color_image, contacts, config.camera
            )
            overlay_img.save(overlay_path)

            entry["grasps"][strategy_name] = {
                "json": json_path,
                "npz": npz_path,
                "overlay": overlay_path,
                "n_contacts": len(contacts),
                "fingers": {
                    label: sum(1 for c in contacts if c.finger == label)
                    for label in set(c.finger for c in contacts)
                },
            }

            print(f"    ✅ {len(contacts)} contact points generated")

        # Save mesh metadata
        meta_path = os.path.join(out_dir, "metadata.json")
        with open(meta_path, "w") as f:
            json.dump(entry, f, indent=2)

        dataset_index.append(entry)

    # Save global index
    index_path = os.path.join(config.output_dir, "dataset_index.json")
    with open(index_path, "w") as f:
        json.dump({
            "n_objects": len(dataset_index),
            "strategies": [s.value for s in config.grasp.strategies],
            "camera": {
                "position": list(config.camera.position),
                "target": list(config.camera.target),
                "resolution": [config.camera.width, config.camera.height],
                "fov": config.camera.fov,
            },
            "objects": dataset_index,
        }, f, indent=2)

    print(f"\n🎉 Dataset generation complete!")
    print(f"   {len(dataset_index)} objects processed")
    print(f"   Output: {config.output_dir}/")
    print(f"   Index:  {index_path}")

#!/usr/bin/env python3
"""
Create sample GLB test objects for the pipeline.
Generates simple parametric shapes so you can test without external data.
"""
import trimesh
import numpy as np
import os


def create_sample_objects(output_dir: str = "data/glb"):
    """Create a few simple GLB objects for testing."""
    os.makedirs(output_dir, exist_ok=True)

    # 1. Cylinder (like a cup without handle)
    cylinder = trimesh.creation.cylinder(radius=0.04, height=0.1, sections=32)
    cylinder.export(os.path.join(output_dir, "cylinder.glb"))
    print("  ✅ cylinder.glb")

    # 2. Sphere
    sphere = trimesh.creation.icosphere(subdivisions=3, radius=0.05)
    sphere.export(os.path.join(output_dir, "sphere.glb"))
    print("  ✅ sphere.glb")

    # 3. Box (like a rectangular object)
    box = trimesh.creation.box(extents=[0.06, 0.04, 0.1])
    box.export(os.path.join(output_dir, "box.glb"))
    print("  ✅ box.glb")

    # 4. Capsule (like a bottle)
    capsule = trimesh.creation.capsule(height=0.08, radius=0.03, count=[16, 16])
    capsule.export(os.path.join(output_dir, "capsule.glb"))
    print("  ✅ capsule.glb")

    # 5. Cone
    cone = trimesh.creation.cone(radius=0.04, height=0.1, sections=32)
    cone.export(os.path.join(output_dir, "cone.glb"))
    print("  ✅ cone.glb")

    print(f"\n📦 {5} test objects created in {output_dir}/")


if __name__ == "__main__":
    create_sample_objects()

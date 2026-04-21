#!/usr/bin/env python3
"""
Visualize grasp contact points in 3D.

Modes:
  1. Single object:  --glb <path> --contacts <path>
  2. All objects:    --all [--strategy front_back]  (objects laid out side by side)

Controls (trimesh viewer):
  - Left-click drag  : rotate
  - Right-click drag : pan
  - Scroll           : zoom
  - Z                : reset view
  - Q / Esc          : quit
"""
import argparse
import glob
import json
import os
import sys
import numpy as np
import trimesh
import pyglet
from PIL import Image, ImageDraw, ImageFont

from grasp_dataset_gen.utils import load_glb, normalize_mesh

# ──────────────────────────────────────────────────────────────────
# Palette
# ──────────────────────────────────────────────────────────────────
FINGER_COLORS = {
    "thumb":  (139, 69,  19),
    "index":  (255, 235,  0),
    "middle": (255, 165,  0),
    "ring":   (230,  30, 30),
    "pinky":  (  0, 190,  0),
    "palm":   ( 50,  50, 50),
}
FINGER_NAMES_FR = {
    "thumb":  "Pouce",
    "index":  "Index",
    "middle": "Majeur",
    "ring":   "Annulaire",
    "pinky":  "Auriculaire",
    "palm":   "Paume",
}


# ──────────────────────────────────────────────────────────────────
# 2D Legend (PIL → OpenGL pixels)
# ──────────────────────────────────────────────────────────────────

def _make_legend_image(strategy: str = "") -> np.ndarray:
    """Build RGBA legend image, flipped vertically for glDrawPixels."""
    pad   = 10; box_w = 16; box_h = 14; row_h = 24; text_x = pad + box_w + 8
    img_w = 195
    img_h = pad + row_h + (len(FINGER_COLORS) + 2) * row_h + pad

    img  = Image.new("RGBA", (img_w, img_h), (20, 20, 20, 210))
    draw = ImageDraw.Draw(img)

    try:
        font_b = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 13)
        font   = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 12)
    except Exception:
        font_b = font = ImageFont.load_default()

    title = f"DOIGTS  [{strategy}]" if strategy else "DOIGTS"
    draw.text((pad, pad), title, fill=(255, 255, 255, 255), font=font_b)

    y = pad + row_h
    for label, fr_name in FINGER_NAMES_FR.items():
        r, g, b = FINGER_COLORS[label]
        draw.rectangle([pad, y+2, pad+box_w, y+2+box_h], fill=(r, g, b, 255),
                       outline=(255, 255, 255, 130))
        draw.text((text_x, y+2), fr_name, fill=(240, 240, 240, 255), font=font)
        y += row_h

    draw.line([(pad, y), (img_w - pad, y)], fill=(120, 120, 120, 200), width=1)
    y += 6

    draw.rectangle([pad, y+2, pad+box_w, y+2+box_h], fill=(240, 240, 240, 255))
    draw.text((text_x, y+2), "Normale", fill=(240, 240, 240, 255), font=font)
    y += row_h
    draw.rectangle([pad, y+2, pad+box_w, y+2+box_h], fill=(0, 220, 220, 255))
    draw.text((text_x, y+2), "Tangente", fill=(240, 240, 240, 255), font=font)

    return np.flipud(np.array(img, dtype=np.uint8))


# ──────────────────────────────────────────────────────────────────
# Custom pyglet viewer (inherits trimesh's SceneViewer)
# ──────────────────────────────────────────────────────────────────

def _make_viewer_class():
    from trimesh.viewer.windowed import SceneViewer
    from OpenGL.GL import (glWindowPos2i, glDrawPixels, GL_RGBA, GL_UNSIGNED_BYTE,
                           glDisable, glEnable, GL_DEPTH_TEST)

    class LegendViewer(SceneViewer):
        def set_legend(self, legend_rgba: np.ndarray):
            self._legend_h, self._legend_w = legend_rgba.shape[:2]
            self._legend_bytes = legend_rgba.tobytes()

        def on_draw(self):
            super().on_draw()
            if not hasattr(self, "_legend_bytes"):
                return
            glDisable(GL_DEPTH_TEST)
            glWindowPos2i(10, 10)
            glDrawPixels(self._legend_w, self._legend_h,
                         GL_RGBA, GL_UNSIGNED_BYTE, self._legend_bytes)
            glEnable(GL_DEPTH_TEST)

    return LegendViewer


# ──────────────────────────────────────────────────────────────────
# Scene builder helpers
# ──────────────────────────────────────────────────────────────────

def _add_contacts_to_scene(scene: trimesh.Scene,
                            contacts: list,
                            offset: np.ndarray = np.zeros(3)):
    """Add contact spheres, normal sticks and tangent sticks to the scene."""
    for contact in contacts:
        pos     = np.array(contact["position"]) + offset
        normal  = np.array(contact["normal"])
        tangent = np.array(contact["tangent"])
        finger  = contact["finger"]
        r, g, b = FINGER_COLORS.get(finger, (200, 200, 200))

        sph = trimesh.creation.uv_sphere(radius=0.004, count=[10, 10])
        sph.apply_translation(pos)
        sph.visual.face_colors = [r, g, b, 255]
        scene.add_geometry(sph)

        if np.linalg.norm(normal) > 1e-6:
            n_end = pos + normal / np.linalg.norm(normal) * 0.02
            st = trimesh.creation.cylinder(radius=0.0007, segment=[pos, n_end])
            st.visual.face_colors = [245, 245, 245, 200]
            scene.add_geometry(st)

        if np.linalg.norm(tangent) > 1e-6:
            t_end = pos + tangent / np.linalg.norm(tangent) * 0.013
            st = trimesh.creation.cylinder(radius=0.0005, segment=[pos, t_end])
            st.visual.face_colors = [0, 210, 210, 180]
            scene.add_geometry(st)


def _launch(scene: trimesh.Scene, strategy: str, title: str):
    """Launch the 3D viewer with legend overlay."""
    legend_rgba = _make_legend_image(strategy)
    LegendViewer = _make_viewer_class()
    viewer = LegendViewer(
        scene,
        resolution=(1100, 750),
        window_title=title,
        start_loop=False,
        visible=True,
    )
    viewer.set_legend(legend_rgba)
    pyglet.app.run()


# ──────────────────────────────────────────────────────────────────
# Single object mode
# ──────────────────────────────────────────────────────────────────

def visualize_grasp(mesh_path: str, contacts_json: str):
    mesh = load_glb(mesh_path)
    mesh = normalize_mesh(mesh)
    mesh.visual.face_colors = [180, 180, 190, 150]

    with open(contacts_json) as f:
        data = json.load(f)

    mesh_name = data.get("mesh", "?")
    strategy  = data.get("strategy", "?")
    print(f"\n  Objet : {mesh_name}   Stratégie : {strategy}   ({data['n_contacts']} pts)")

    scene = trimesh.Scene([mesh])
    _add_contacts_to_scene(scene, data["contacts"])

    _launch(scene, strategy, f"Grasp — {mesh_name} [{strategy}]")


# ──────────────────────────────────────────────────────────────────
# All-objects mode
# ──────────────────────────────────────────────────────────────────

def visualize_all(glb_dir: str, output_dir: str, strategy: str):
    """
    Load every object from glb_dir and show them side by side
    in one scene, each with their contact points for `strategy`.
    """
    glb_files = sorted(glob.glob(os.path.join(glb_dir, "*.glb")))
    if not glb_files:
        print(f"No .glb files found in {glb_dir}")
        return

    scene   = trimesh.Scene()
    spacing = 0.26    # meters between object centers
    loaded  = 0

    for i, glb_path in enumerate(glb_files):
        mesh_name     = os.path.splitext(os.path.basename(glb_path))[0]
        contacts_path = os.path.join(output_dir, mesh_name, f"grasp_{strategy}.json")

        if not os.path.exists(contacts_path):
            print(f"  ⚠️  {mesh_name}: no contacts file ({strategy}), skipping")
            continue

        mesh = load_glb(glb_path)
        mesh = normalize_mesh(mesh)

        # Layout: row of objects along X axis
        offset = np.array([i * spacing, 0.0, 0.0])
        mesh.vertices += offset
        mesh.visual.face_colors = [170 + i * 12 % 60, 175, 185, 140]
        scene.add_geometry(mesh)

        with open(contacts_path) as f:
            data = json.load(f)

        _add_contacts_to_scene(scene, data["contacts"], offset=offset)

        print(f"  ✓  {mesh_name:20s} — {data['n_contacts']} points")
        loaded += 1

    if loaded == 0:
        print("Nothing to display.")
        return

    print(f"\n  {loaded} objects loaded, strategy = {strategy}")
    _launch(scene, strategy, f"Grasp — ALL objects [{strategy}]")


# ──────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="3D grasp visualizer with legend overlay",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single object:
  python visualize_3d.py --glb data/glb/cylinder.glb --contacts output/cylinder/grasp_front_back.json

  # All objects side by side:
  python visualize_3d.py --all
  python visualize_3d.py --all --strategy left_right
""",
    )
    parser.add_argument("--all",      action="store_true", help="Show all objects side by side")
    parser.add_argument("--strategy", default="front_back",
                        choices=["front_back", "left_right", "right_left"],
                        help="Strategy to display in --all mode (default: front_back)")
    parser.add_argument("--glb_dir",    default="data/glb",  help="GLB directory (--all mode)")
    parser.add_argument("--output_dir", default="output",    help="Output directory (--all mode)")
    parser.add_argument("--glb",      help="GLB file (single mode)")
    parser.add_argument("--contacts", help="Contacts JSON file (single mode)")
    args = parser.parse_args()

    if args.all:
        visualize_all(args.glb_dir, args.output_dir, args.strategy)
    elif args.glb and args.contacts:
        visualize_grasp(args.glb, args.contacts)
    else:
        parser.error("Specify either --all or both --glb and --contacts")


if __name__ == "__main__":
    main()

import objaverse
import os
import shutil
import trimesh
import argparse

def download_subset(num_per_cat=10):
    # Filter objects from LVIS taxonomy (high quality)
    print("⏳ Loading Objaverse-LVIS annotations...")
    lvis_annotations = objaverse.load_lvis_annotations()

    # Select categories suitable for manipulation
    categories = ["cup", "bottle", "hammer", "screwdriver", "wrench"]
    
    target_dir = "data/objaverse"
    os.makedirs(target_dir, exist_ok=True)

    for cat in categories:
        if cat not in lvis_annotations:
            continue
            
        print(f"🔍 Processing category: {cat}...")
        # Get more UIDs than needed to filter multi-object scenes
        pool_uids = lvis_annotations[cat][:num_per_cat * 3]
        objects = objaverse.load_objects(uids=pool_uids)
        
        count = 0
        for uid, path in objects.items():
            if count >= num_per_cat:
                break
                
            mesh = trimesh.load(path, force='mesh')
            if isinstance(mesh, trimesh.Scene):
                mesh = mesh.dump(concatenate=True)
            
            # Simple multi-object filter: check connected components
            # If the largest component is not significantly dominant, it's a "scene"
            comp = mesh.split()
            if len(comp) > 1:
                # Sort by number of vertices
                comp = sorted(comp, key=lambda x: len(x.vertices), reverse=True)
                # If second largest part has more than 5% vertices, it's likely multiple objects
                if len(comp[1].vertices) > (len(mesh.vertices) * 0.05):
                    print(f"  ⚠️  Skipping {uid} (detected multiple objects)")
                    continue
            
            count += 1
            dest = os.path.join(target_dir, f"{cat}_{count:02d}_{uid[:6]}.glb")
            shutil.copy(path, dest)
            print(f"  ✅ Saved {cat} {count}/{num_per_cat}: {dest}")

if __name__ == "__main__":
    # Clean data/objaverse to avoid mixing old/new naming
    if os.path.exists("data/objaverse"):
        shutil.rmtree("data/objaverse")
    download_subset()

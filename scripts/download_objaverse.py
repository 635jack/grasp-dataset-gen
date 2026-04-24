import objaverse
import os
import shutil
import argparse

def download_subset(num_per_cat=10):
    # Filter objects from LVIS taxonomy (high quality)
    print("⏳ Loading Objaverse-LVIS annotations...")
    lvis_annotations = objaverse.load_lvis_annotations()

    # Select categories suitable for manipulation
    categories = ["cup", "bottle", "hammer", "screwdriver", "wrench"]
    uids = []
    
    for cat in categories:
        if cat in lvis_annotations:
            # Take the first N of each category
            cat_uids = lvis_annotations[cat][:num_per_cat]
            uids.extend(cat_uids)
            print(f"✅ Selected {len(cat_uids)} objects for category: {cat}")

    print(f"📦 Downloading {len(uids)} objects from Objaverse...")
    # This might take a bit of time depending on bandwidth
    objects = objaverse.load_objects(uids=uids)

    # Move to our data/objaverse folder
    target_dir = "data/objaverse"
    os.makedirs(target_dir, exist_ok=True)
    
    for uid, path in objects.items():
        dest = os.path.join(target_dir, f"{uid}.glb")
        shutil.copy(path, dest)
        print(f"  -> Saved {uid} to {dest}")

if __name__ == "__main__":
    download_subset()

import json
import os

def generate_card(index_path, output_path):
    with open(index_path, "r") as f:
        data = json.load(f)
    
    n_objects = data.get("n_objects", 0)
    strategies = data.get("strategies", [])
    
    content = f"""---
language: en
license: mit
task_categories:
- robotics
- computer-vision
tags:
- 3d
- grasp-synthesis
- tactile-sensing
- objaverse
pretty_name: Synthetic Grasp Dataset (Curated)
---

# 🖐️ Synthetic Grasp Dataset (Objaverse-LVIS Curated)

This dataset contains high-quality synthetic grasp data generated for robotic manipulation research. 
It focuses on the fusion of **vision** and **tactile** sensing by providing visibility and occlusion analysis for each contact point.

## 📊 Dataset Statistics
- **Number of objects:** {n_objects}
- **Source:** Curated objects from [Objaverse-LVIS](https://objaverse.allenai.org/) (categories: cup, bottle, hammer, screwdriver, wrench)
- **Grasp Strategies:** {", ".join(strategies)}
- **Camera Resolution:** {data['camera']['resolution'][0]}x{data['camera']['resolution'][1]}

## 🛠️ Data Format

Each object folder contains:
- `rgb.png`: Monocular RGB render.
- `grasp_<strategy>.json`: Contact points with position, normal, tangent, and visibility status.
- `grasp_<strategy>.npz`: NumPy version of the contact points.
- `grasp_<strategy>_overlay.png`: Visual overlay of the grasp on the object.
- `metadata.json`: Object-specific metadata (surface visibility, bounding box, complexity).

## 🔍 Visibility Classification
Every contact point is classified based on camera occlusion:
- **VISIBLE**: Point is directly seen by the camera.
- **SILHOUETTE**: Point is on the visual horizon (critical for tactile تکمیل).
- **OCCLUDED**: Point is hidden by the object itself (back side or self-occlusion).

## 📜 How to use
This dataset is designed to train models that predict contact stability from visual data or to simulate-to-real transfer for tactile controllers.

```python
from huggingface_hub import snapshot_download
path = snapshot_download("jack635/grasp-dataset-curated", repo_type="dataset")
```

---
*Generated using the [Grasp Dataset Generator](https://github.com/635jack/grasp-dataset-gen) pipeline.*
"""
    with open(output_path, "w") as f:
        f.write(content)
    print(f"✅ Dataset Card generated at {output_path}")

if __name__ == "__main__":
    generate_card("output_hf/dataset_index.json", "output_hf/README.md")

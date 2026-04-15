# 🖐️ Grasp Dataset Generator

Génération de datasets de préhension à partir d'objets 3D (GLB).

Pour chaque objet, le pipeline produit :
- **Image RGB monoculaire** : rendu depuis un point de vue fixé
- **Points de contact de préhension** sur la surface 3D, chacun contenant :
  - Position `(x, y, z)` dans le repère monde
  - Normale surfacique (vecteur unitaire sortant du mesh)
  - Tangente surfacique (vecteur unitaire dans le plan tangent, direction de glissement)
  - Label de doigt : `thumb`, `index`, `middle`, `ring`, `pinky`, `palm`

## 🤏 Stratégies de préhension

Trois configurations de prise sont générées par défaut :

| Stratégie | Pouce | 4 doigts | Paume |
|-----------|-------|----------|-------|
| **front_back** | Devant (côté caméra) | Derrière (côté occulté) | Devant |
| **left_right** | Gauche | Droite | Face caméra |
| **right_left** | Droite | Gauche | Face caméra |

## 📁 Structure du projet

```
grasp-dataset-gen/
├── data/glb/                    # ← Placez vos fichiers .glb ici
├── output/                      # ← Résultats générés
│   ├── <objet>/
│   │   ├── rgb.png              # Rendu RGB
│   │   ├── grasp_front_back.json    # Points de contact (JSON)
│   │   ├── grasp_front_back.npz     # Points de contact (NumPy)
│   │   ├── grasp_front_back_overlay.png  # Visualisation
│   │   ├── grasp_left_right.*
│   │   ├── grasp_right_left.*
│   │   └── metadata.json
│   └── dataset_index.json       # Index global
├── grasp_dataset_gen/           # Package Python
│   ├── config.py                # Configuration (caméra, préhension)
│   ├── renderer.py              # Rendu off-screen (pyrender)
│   ├── grasp_sampler.py         # Échantillonnage de points de contact
│   ├── dataset.py               # Orchestrateur
│   └── utils.py                 # I/O et visualisation 2D
├── generate_dataset.py          # Point d'entrée CLI
├── visualize_3d.py              # Visualisation 3D interactive
└── requirements.txt
```

## 🚀 Installation

```bash
cd grasp-dataset-gen
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

> **Note macOS** : Si `pyrender` ne fonctionne pas en headless, essayez :
> ```bash
> pip install PyOpenGL PyOpenGL_accelerate
> # Ou utilisez le backend osmesa :
> export PYOPENGL_PLATFORM=osmesa
> ```

## 💡 Utilisation

### 1. Placer les objets

Copiez vos fichiers `.glb` dans `data/glb/` :
```bash
cp /chemin/vers/mon_objet.glb data/glb/
```

### 2. Générer le dataset

```bash
python generate_dataset.py
```

Options disponibles :
```bash
python generate_dataset.py \
  --glb_dir data/glb \
  --output_dir output \
  --resolution 1024 768 \
  --fov 45 \
  --strategies front_back left_right right_left \
  --points_per_finger 5 \
  --points_palm 8 \
  --seed 42
```

### 3. Visualiser en 3D

```bash
python visualize_3d.py \
  --glb data/glb/mon_objet.glb \
  --contacts output/mon_objet/grasp_front_back.json
```

## 📊 Format des données

### Contact point (JSON)

```json
{
  "mesh": "cup_v1",
  "strategy": "front_back",
  "n_contacts": 33,
  "contacts": [
    {
      "position": [0.012, -0.034, 0.056],
      "normal": [0.0, -0.98, 0.19],
      "tangent": [0.0, -0.19, -0.98],
      "finger": "thumb"
    }
  ]
}
```

### Contact point (NPZ)

```python
import numpy as np
data = np.load("grasp_front_back.npz")
data["positions"]   # (N, 3) float32
data["normals"]     # (N, 3) float32
data["tangents"]    # (N, 3) float32
data["fingers"]     # (N,)   string
```

## 🔧 Configuration avancée

Voir `grasp_dataset_gen/config.py` pour tous les paramètres :

- `CameraConfig` : position, FOV, résolution, clipping planes
- `GraspConfig` : points par doigt, écartement, bruit sur les rayons
- `DatasetConfig` : répertoires, seed, éclairage, couleur de fond

## 📜 Licence

MIT

import streamlit as st
import json
import os
import trimesh
import numpy as np
import base64
from PIL import Image

# Import local utilities
from grasp_dataset_gen.utils import load_glb, normalize_mesh, FINGER_COLORS

st.set_page_config(page_title="Grasp Dataset Viewer", layout="wide")

st.title("🖐️ Grasp Dataset Inspector")
st.markdown("Inspect results for 50+ objects with 3D views and visibility analysis.")

INDEX_PATH = "output_hf/dataset_index.json"

if not os.path.exists(INDEX_PATH):
    st.error(f"Dataset index not found at {INDEX_PATH}. Please run the pipeline first.")
    st.stop()

with open(INDEX_PATH, "r") as f:
    index_data = json.load(f)

# Sidebar
st.sidebar.header("Navigation")
obj_names = [obj["mesh"] for obj in index_data["objects"]]
selected_obj_name = st.sidebar.selectbox("Select Object", obj_names)

selected_obj = next(obj for obj in index_data["objects"] if obj["mesh"] == selected_obj_name)
available_strategies = list(selected_obj["grasps"].keys())
selected_strategy = st.sidebar.radio("Strategy", available_strategies)

# Metrics
st.sidebar.markdown("---")
st.sidebar.markdown(f"**Object Metadata**")
st.sidebar.json(selected_obj.get("metadata", {}))

# Main Content
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📷 2D Overlay")
    overlay_path = selected_obj["grasps"][selected_strategy]["overlay"].replace("output/", "output_hf/", 1)
    if os.path.exists(overlay_path):
        st.image(Image.open(overlay_path), use_container_width=True)
        st.caption(f"Overlay for {selected_strategy} (X = Occluded)")
    else:
        st.warning("Overlay image missing.")

with col2:
    st.subheader("🧊 3D Contact View")
    
    # Generate temporary 3D scene with sticks for normals and TANGENTS
    mesh_path = os.path.join("data/objaverse", f"{selected_obj_name}.glb")
    if os.path.exists(mesh_path):
        mesh = load_glb(mesh_path)
        mesh = normalize_mesh(mesh)
        
        # Load contact data
        json_path = selected_obj["grasps"][selected_strategy]["json"].replace("output/", "output_hf/", 1)
        with open(json_path, "r") as fj:
            cdata = json.load(fj)
        
        scene = trimesh.Scene()
        # Semi-transparent mesh
        m = mesh.copy()
        m.visual.face_colors = [200, 200, 210, 150]
        scene.add_geometry(m)
        
        for contact in cdata["contacts"]:
            pos = np.array(contact["position"])
            norm = np.array(contact["normal"])
            tang = np.array(contact["tangent"])
            color = FINGER_COLORS.get(contact["finger"], (200, 200, 200))
            
            # Sphere
            sph = trimesh.creation.uv_sphere(radius=0.002, count=[10, 10])
            sph.apply_translation(pos)
            sph.visual.face_colors = list(color) + [255]
            scene.add_geometry(sph)
            
            # Normal (White)
            n_end = pos + norm * 0.015
            sn = trimesh.creation.cylinder(radius=0.0004, segment=[pos, n_end])
            sn.visual.face_colors = [255, 255, 255, 200]
            scene.add_geometry(sn)
            
            # Tangent (Cyan) - This is what the user wants to check!
            t_end = pos + tang * 0.015
            st_cyl = trimesh.creation.cylinder(radius=0.0004, segment=[pos, t_end])
            st_cyl.visual.face_colors = [0, 255, 255, 255]
            scene.add_geometry(st_cyl)

        # Export to temporary GLB bytes
        glb_data = scene.export(file_type="glb")
        b64_glb = base64.b64encode(glb_data).decode()
        
        # HTML for model-viewer
        html = f"""
            <script type="module" src="https://ajax.googleapis.com/ajax/libs/model-viewer/3.3.0/model-viewer.min.js"></script>
            <model-viewer 
                src="data:model/gltf-binary;base64,{b64_glb}" 
                style="width: 100%; height: 500px; background-color: #111;" 
                auto-rotate camera-controls shadow-intensity="1">
            </model-viewer>
        """
        st.components.v1.html(html, height=550)
        st.info("⚪ Lignes blanches : Normales | 🔵 Lignes turquoises : Tangentes")
    else:
        st.error(f"3D Mesh not found at {mesh_path}")

st.markdown("---")
st.subheader("📄 Raw Contact Data")
st.dataframe(cdata["contacts"])

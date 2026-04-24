#!/usr/bin/env python3
"""
Automatically generate a LaTeX report summarizing the grasp dataset results.
Updated with color legends and a full data appendix.
"""
import json
import os
import datetime
import csv

def tex_escape(text):
    """Escape LaTeX special characters."""
    if not isinstance(text, str):
        return str(text)
    return text.replace("_", "\\_").replace("#", "\\#").replace("%", "\\%").replace("&", "\\&")

FINGER_COLORS = {
    "thumb":  (139, 69, 19),
    "index":  (255, 255, 0),
    "middle": (255, 165, 0),
    "ring":   (255, 0, 0),
    "pinky":  (0, 255, 0),
    "palm":   (0, 0, 0),
}

FINGER_NAMES_FR = {
    "thumb":  "Pouce",
    "index":  "Index",
    "middle": "Majeur",
    "ring":   "Annulaire",
    "pinky":  "Auriculaire",
    "palm":   "Paume",
}

def generate_report(index_path="output/dataset_index.json", csv_path="output/grasp_dataset.csv", output_tex="output/rapport_dataset.tex"):
    if not os.path.exists(index_path):
        print(f"Error: {index_path} not found.")
        return

    with open(index_path, 'r') as f:
        data = json.load(f)

    date_str = datetime.datetime.now().strftime("%d %B %Y à %H:%M")
    
    # --- Start of LaTeX Document ---
    tex = [
        r"\documentclass[a4paper,11pt]{article}",
        r"\usepackage[utf8]{inputenc}",
        r"\usepackage[T1]{fontenc}",
        r"\usepackage[french]{babel}",
        r"\usepackage{graphicx}",
        r"\usepackage{geometry}",
        r"\usepackage{booktabs}",
        r"\usepackage{subcaption}",
        r"\usepackage{float}",
        r"\usepackage{hyperref}",
        r"\usepackage{url}",
        r"\usepackage{xcolor}",
        r"\usepackage{longtable}",
        r"\usepackage{array}",
        r"\definecolor{visvisible}{RGB}{0, 150, 0}",
        r"\definecolor{vissilhouette}{RGB}{200, 100, 0}",
        r"\definecolor{visoccluded}{RGB}{200, 0, 0}",
    ]
    
    # Define finger colors
    for name, (r, g, b) in FINGER_COLORS.items():
        tex.append(f"\\definecolor{{color{name}}}{{RGB}}{{{r}, {g}, {b}}}")
    
    tex += [
        r"\definecolor{navy}{RGB}{20, 40, 80}",
        r"\geometry{margin=2cm}",
        r"\title{\color{navy}\bfseries Rapport de Génération de Dataset de Saisie Robotique}",
        r"\author{Système de Génération Automatique (ISIR)}",
        f"\\date{{{date_str}}}",
        r"\begin{document}",
        r"\maketitle",
        r"\tableofcontents",
        r"\newpage",
        
        r"\section{Résumé de la Configuration}",
        r"Ce rapport présente les résultats de la génération synthétique de points de saisie 6-DOF basée sur un modèle de cylindre virtuel.",
        r"\begin{itemize}",
        f"    \\item \\textbf{{Nombre d'objets :}} {data['n_objects']}",
        f"    \\item \\textbf{{Stratégies évaluées :}} {', '.join([tex_escape(s) for s in data['strategies']])}",
        f"    \\item \\textbf{{Champ de vision (FOV) :}} {data['camera']['fov']}$^\circ$",
        f"    \\item \\textbf{{Résolution :}} {data['camera']['resolution'][0]}x{data['camera']['resolution'][1]} pixels",
        r"\end{itemize}",
        
        r"\subsection{Légende des Couleurs}",
        r"Chaque doigt est représenté par une couleur spécifique dans les rendus 2D et 3D :",
        r"\begin{center}",
        r"\begin{tabular}{ll@{\hspace{2em}}ll}",
    ]
    
    # Two columns for the legend
    fingers = list(FINGER_COLORS.keys())
    for i in range(0, len(fingers), 2):
        f1 = fingers[i]
        f2 = fingers[i+1] if i+1 < len(fingers) else None
        
        row = f"\\textcolor{{color{f1}}}{{\\rule{{10pt}}{{10pt}}}} & {FINGER_NAMES_FR[f1]}"
        if f2:
            row += f" & \\textcolor{{color{f2}}}{{\\rule{{10pt}}{{10pt}}}} & {FINGER_NAMES_FR[f2]} \\\\"
        else:
            row += " & & \\\\"
        tex.append(row)
        
    tex += [
        r"\end{tabular}",
        r"\end{center}",
        r"\vspace{1em}",
    ]

    # --- Iterate per Object ---
    for obj in data.get("objects", []):
        mesh_name = obj["mesh"]
        tex.append(f"\\section{{Objet : {tex_escape(mesh_name)}}}")
        
        # Meta info
        tex.append(r"\begin{tabular}{ll}")
        tex.append(f"    \\textbf{{Complexité :}} & {obj['n_vertices']} sommets, {obj['n_faces']} faces \\\\")
        bb = obj["bounding_box"]
        dim = [abs(bb[1][i] - bb[0][i]) * 100 for i in range(3)] # cm
        tex.append(f"    \\textbf{{Dimensions (cm) :}} & {dim[0]:.1f} x {dim[1]:.1f} x {dim[2]:.1f} \\\\")
        
        # New: Surface Visibility
        s_vis = obj.get("surface_visibility", 0)
        tex.append(f"    \\textbf{{Visibilité Surface :}} & {s_vis:.1%} \\\\")
        
        tex.append(r"\end{tabular}")
        tex.append(r"\vspace{1em}")

        # Image Gallery
        tex.append(r"\begin{figure}[H]")
        tex.append(r"    \centering")
        
        # RGB (Top left) - Always present
        rgb_path = obj['rgb'].replace('output/', '', 1).replace('output_hf/', '', 1)
        tex.append(r"    \begin{subfigure}[b]{0.45\textwidth}")
        tex.append(f"        \\includegraphics[width=\\textwidth]{{{rgb_path}}}")
        tex.append(r"        \caption{Rendu RGB brut}")
        tex.append(r"    \end{subfigure}")
        tex.append(r"    \hfill")
        
        # Front Back (Top right)
        if "front_back" in obj["grasps"]:
            fb_img = obj["grasps"]["front_back"]["overlay"].replace('output/', '', 1).replace('output_hf/', '', 1)
            tex.append(r"    \begin{subfigure}[b]{0.45\textwidth}")
            tex.append(f"        \\includegraphics[width=\\textwidth]{{{fb_img}}}")
            tex.append(r"        \caption{Stratégie Front-Back}")
            tex.append(r"    \end{subfigure}")
        
        tex.append(r"    \\[1ex]") # Line break
        
        # Left Right (Bottom left)
        if "left_right" in obj["grasps"]:
            lr_img = obj["grasps"]["left_right"]["overlay"].replace('output/', '', 1).replace('output_hf/', '', 1)
            tex.append(r"    \begin{subfigure}[b]{0.45\textwidth}")
            tex.append(f"        \\includegraphics[width=\\textwidth]{{{lr_img}}}")
            tex.append(r"        \caption{Stratégie Left-Right}")
            tex.append(r"    \end{subfigure}")
            tex.append(r"    \hfill")
        
        # Right Left (Bottom right)
        if "right_left" in obj["grasps"]:
            rl_img = obj["grasps"]["right_left"]["overlay"].replace('output/', '', 1).replace('output_hf/', '', 1)
            tex.append(r"    \begin{subfigure}[b]{0.45\textwidth}")
            tex.append(f"        \\includegraphics[width=\\textwidth]{{{rl_img}}}")
            tex.append(r"        \caption{Stratégie Right-Left}")
            tex.append(r"    \end{subfigure}")
        
        tex.append(f"    \\caption{{Vues de l'objet \\textbf{{{tex_escape(mesh_name)}}}}}")
        tex.append(r"\end{figure}")
        
        # Contact Summary Table
        tex.append(r"\vspace{1em}")
        tex.append(r"\begin{center}")
        tex.append(r"\begin{tabular}{@{}lcccc@{}} \toprule")
        tex.append(r"Stratégie & Nb Contacts & Visibilité & Doigts concernés & Fichier \\ \midrule")
        for strat, info in obj["grasps"].items():
            # Load JSON to count visibility
            json_path_abs = os.path.join(os.path.dirname(index_path), info['json'].replace('output/', '', 1).replace('output_hf/', '', 1))
            visible_count = 0
            total_count = 0
            try:
                with open(json_path_abs, 'r') as fj:
                    cdata = json.load(fj)
                    for c in cdata.get("contacts", []):
                        total_count += 1
                        if c.get("visibility") == "VISIBLE":
                            visible_count += 1
            except:
                pass
            
            vis_ratio = f"{visible_count}/{total_count}" if total_count > 0 else "N/A"
            color = "visvisible" if visible_count == total_count else ("visoccluded" if visible_count == 0 else "vissilhouette")
            vis_tex = f"\\textbf{{\\color{{{color}}}{vis_ratio}}}"
            
            fingers_list = [FINGER_NAMES_FR.get(f, f) for f in info["fingers"].keys()]
            fingers_str = ", ".join(fingers_list)
            json_path = info['json'].replace('output/', '', 1).replace('output_hf/', '', 1)
            tex.append(f"{tex_escape(strat)} & {info['n_contacts']} & {vis_tex} & \\small {tex_escape(fingers_str)} & \\path{{{tex_escape(json_path)}}} \\\\")
        tex.append(r"\bottomrule \end{tabular}")
        tex.append(r"\end{center}")
        
        tex.append(r"\newpage")

    # --- Appendix: Complete Data ---
    if os.path.exists(csv_path):
        tex += [
            r"\section{Annexe : Données de Saisie Complètes}",
            r"Ce tableau récapitule l'ensemble des points de contact générés pour tous les objets et toutes les stratégies, incluant le statut de visibilité.",
            r"\begin{center}",
            r"\small",
            r"\begin{longtable}{llllc}",
            r"\caption{Détail complet des points de contact} \\ \toprule",
            r"Objet & Stratégie & Doigt & Statut & Position (x, y, z) \\ \midrule",
            r"\endfirsthead",
            r"\toprule Objet & Stratégie & Doigt & Statut & Position (x, y, z) \\ \midrule",
            r"\endhead",
            r"\bottomrule",
            r"\endfoot",
        ]
        
        with open(csv_path, 'r') as f_csv:
            reader = csv.DictReader(f_csv)
            for row in reader:
                mesh = tex_escape(row['mesh'])
                strat = tex_escape(row['strategy'])
                finger = tex_escape(FINGER_NAMES_FR.get(row['finger'], row['finger']))
                
                # Apply color to visibility status
                raw_vis = row.get('visibility', 'UNKNOWN')
                if "VISIBLE" in raw_vis:
                    vis_fmt = f"\\textbf{{\\color{{visvisible}}{tex_escape(raw_vis)}}}"
                elif "SILHOUETTE" in raw_vis:
                    vis_fmt = f"\\textbf{{\\color{{vissilhouette}}{tex_escape(raw_vis)}}}"
                else:
                    vis_fmt = f"\\textbf{{\\color{{visoccluded}}{tex_escape(raw_vis)}}}"
                
                # Use simplified position to avoid table overflow
                pos = f"\\small ({float(row['pos_x']):.3f}, {float(row['pos_y']):.3f}, {float(row['pos_z']):.3f})"
                
                tex.append(f"{mesh} & {strat} & {finger} & {vis_fmt} & {pos} \\\\")
        
        tex += [
            r"\end{longtable}",
            r"\end{center}",
        ]

    # --- End of Document ---
    tex.append(r"\end{document}")

    # Write to file
    with open(output_tex, 'w', encoding='utf-8') as f:
        f.write("\n".join(tex))

    print(f"✅ Rapport LaTeX complet généré : {output_tex}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", default="output/dataset_index.json", help="Path to input JSON index")
    parser.add_argument("--csv", default="output/grasp_dataset.csv", help="Path to input CSV (for appendix)")
    parser.add_argument("--output", default="output/rapport_dataset.tex", help="Path to output .tex file")
    args = parser.parse_args()
    
    generate_report(args.index, args.csv, args.output)

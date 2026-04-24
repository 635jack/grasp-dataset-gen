import json
import csv
import os

def export_dataset_to_csv(index_path, csv_output_path):
    if not os.path.exists(index_path):
        print(f"Index file not found: {index_path}")
        return

    with open(index_path, 'r') as f:
        index_data = json.load(f)

    all_rows = []
    
    # Headers
    headers = [
        "mesh", "strategy", "finger", "visibility",
        "pos_x", "pos_y", "pos_z",
        "norm_x", "norm_y", "norm_z",
        "tang_x", "tang_y", "tang_z"
    ]

    for obj in index_data.get("objects", []):
        mesh_name = obj.get("mesh", "unknown")
        for strategy_name, info in obj.get("grasps", {}).items():
            json_path = info.get("json")
            if not json_path or not os.path.exists(json_path):
                continue
            
            with open(json_path, 'r') as f_contacts:
                contact_data = json.load(f_contacts)
                
            for c in contact_data.get("contacts", []):
                pos = c["position"]
                norm = c["normal"]
                tang = c["tangent"]
                finger = c["finger"]
                visibility = c.get("visibility", "UNKNOWN")
                
                row = [
                    mesh_name, strategy_name, finger, visibility,
                    pos[0], pos[1], pos[2],
                    norm[0], norm[1], norm[2],
                    tang[0], tang[1], tang[2]
                ]
                all_rows.append(row)

    with open(csv_output_path, 'w', newline='') as f_csv:
        writer = csv.writer(f_csv)
        writer.writerow(headers)
        writer.writerows(all_rows)

    print(f"✅ Dataset exported to {csv_output_path} ({len(all_rows)} rows)")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", default="output/dataset_index.json", help="Path to input JSON index")
    parser.add_argument("--output", default="output/grasp_dataset.csv", help="Path to output CSV")
    args = parser.parse_args()
    
    export_dataset_to_csv(args.index, args.output)

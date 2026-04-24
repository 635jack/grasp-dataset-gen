import os
import argparse
from huggingface_hub import HfApi, create_repo, upload_folder

def upload(folder_path, repo_name):
    api = HfApi()
    username = api.whoami()['name']
    repo_id = f"{username}/{repo_name}"
    
    print(f"🚀 Creating/Checking repository: {repo_id}...")
    try:
        create_repo(repo_id, repo_type="dataset", exist_ok=True)
        print(f"✅ Repository {repo_id} is ready.")
    except Exception as e:
        print(f"⚠️  Repository creation info: {e}")

    print(f"📤 Uploading folder '{folder_path}' to {repo_id}...")
    # This will upload the entire folder (images, CSV, report)
    api.upload_folder(
        folder_path=folder_path,
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="Initial upload of synthetic grasp dataset",
    )
    print(f"🎉 Upload complete! View it at: https://huggingface.co/datasets/{repo_id}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", default="output_hf", help="Folder to upload")
    parser.add_argument("--repo", default="grasp-dataset-curated", help="Name of the HF repository")
    args = parser.parse_args()
    
    upload(args.folder, args.repo)

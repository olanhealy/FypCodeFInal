import os
import json
import subprocess
from tqdm import tqdm  

# Directory to store cloned repositories above my git repository 
CLONE_DIR = os.path.abspath("../../../repos")  

# Base URL for github repositories
BASE_REPO_URL = "https://github.com/"

# Function to clone a repository
def clone_repo(repo_name, destination_dir):
    """
    Clones a repository into the repos directory if it doesn't already exist.
    """

    repo_url = os.path.join(BASE_REPO_URL, repo_name.replace(".", "/"))
    clone_path = os.path.join(destination_dir, repo_name)
    
    if not os.path.exists(clone_path):
        print(f"Cloning {repo_url} into {clone_path}...")
        try:
            subprocess.run(
                ["git", "clone", repo_url, clone_path],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            print(f"Successfully cloned {repo_name}")
        except subprocess.CalledProcessError as e:
            print(f"Failed to clone {repo_name}: {e.stderr.decode().strip()}")
    else:
        print(f"Repository {repo_name} already exists at {clone_path}. Skipping.")

# Load dataset 
dataset_path = "../../Data/sstubs4j/unique/sstubsLarge-train.json"  
if not os.path.exists(dataset_path):
    print(f"Dataset file not found at {dataset_path}")
    exit(1)

with open(dataset_path, "r") as file:
    dataset = json.load(file)

# Create the directory for cloned repositories if it doesn't exist
os.makedirs(CLONE_DIR, exist_ok=True)

# Extract repository names and clone
repo_names = set(bug.get("projectName") for bug in dataset if "projectName" in bug)
repo_names = [repo for repo in repo_names if repo]  # Ensure no None or empty names

# progress bar
for repo_name in tqdm(repo_names, desc="Cloning repositories"):
    clone_repo(repo_name, CLONE_DIR)

print("All repositories processed.")


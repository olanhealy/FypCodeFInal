import os
import json
import subprocess
from tqdm import tqdm

# Relevant  directories
DATA_DIR = "../../Data/sstubs4j/repetition"
REPO_BASE_DIR = "../../../repos"
OUTPUT_DIR = os.path.join(DATA_DIR, "enhanced")

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Helper function to run a git command (for getting commit messages)
def run_git_command(repo_path, command):
    try:
        result = subprocess.run(command, cwd=repo_path, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return result.stdout.decode().strip()
    except subprocess.CalledProcessError as e:
        print(f"Git command failed: {e.stderr.decode().strip()}")
        return None

# Get commit message for a given SHA1
def get_commit_message(repo_path, commit_sha):
    return run_git_command(repo_path, ["git", "show", "-s", "--format=%s", commit_sha])

# Extract context from the `fixPatch` field in dataset 
def extract_context_from_patch(fix_patch, context_lines=5):
    lines = fix_patch.split("\n")
    context_before, buggy_code, context_after = "", "", ""

    for i, line in enumerate(lines):
        if line.startswith("-") and not line.startswith("---"):
            # Found buggy line
            buggy_code = line[1:].strip()
            start = max(0, i - context_lines)
            end = min(len(lines), i + context_lines + 1)

            context_before = "\n".join(
                [l[1:].strip() for l in lines[start:i] if l.startswith("-") or l.startswith(" ")]
            )
            context_after = "\n".join(
                [l[1:].strip() for l in lines[i + 1:end] if l.startswith("+") or l.startswith(" ")]
            )
            break

    return context_before, buggy_code, context_after

# Process a single dataset file
def process_dataset_file(dataset_path):
    with open(dataset_path, "r") as file:
        dataset = json.load(file)

    enhanced_dataset = []

    for bug in tqdm(dataset, desc=f"Processing {os.path.basename(dataset_path)}"):
        repo_name = bug.get("projectName")
        repo_path = os.path.join(REPO_BASE_DIR, repo_name)

        if not os.path.exists(repo_path):
            print(f"Repository {repo_name} not found. Skipping bug.")
            continue

        fix_commit = bug.get("fixCommitSHA1")
        parent_commit = bug.get("fixCommitParentSHA1")

        # Get commit messages
        bug["fixCommitMessage"] = get_commit_message(repo_path, fix_commit) or ""
        bug["parentCommitMessage"] = get_commit_message(repo_path, parent_commit) or ""

        # Extract the context
        fix_patch = bug.get("fixPatch", "")
        context_before, buggy_code, context_after = extract_context_from_patch(fix_patch)

        bug["contextBefore"] = context_before
        bug["buggyCode"] = buggy_code
        bug["contextAfter"] = context_after

        enhanced_dataset.append(bug)

    # Save the enhanced dataset
    output_file = os.path.join(OUTPUT_DIR, os.path.basename(dataset_path))
    with open(output_file, "w") as file:
        json.dump(enhanced_dataset, file, indent=4)

# Process all dataset files in the unique folder
def process_all_datasets():
    for dataset_file in ["sstubsLarge-train.json", "sstubsLarge-test.json", "sstubsLarge-val.json"]:
        dataset_path = os.path.join(DATA_DIR, dataset_file)
        if os.path.exists(dataset_path):
            process_dataset_file(dataset_path)
        else:
            print(f"Dataset file {dataset_path} not found. Skipping.")

if __name__ == "__main__":
    process_all_datasets()
    print("All datasets have been processed and saved to the enhanced folder.")

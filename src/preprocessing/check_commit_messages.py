import os
import json

def check_commit_messages(file_path):
    """
    Check if all bugs in the JSON file have non-empty fixCommitMessage and parentCommitMessage.
    """
    try:
        print(f"Processing file: {file_path}")
        with open(file_path, 'r') as file:
            data = json.load(file)

        total_bugs = len(data)
        missing_fix_commit_message = sum(1 for item in data if not item.get("fixCommitMessage"))
        missing_parent_commit_message = sum(1 for item in data if not item.get("parentCommitMessage"))

        print(f"\nResults for {file_path}:")
        print(f"  Total bugs: {total_bugs}")
        print(f"  Bugs missing fixCommitMessage: {missing_fix_commit_message}")
        print(f"  Bugs missing parentCommitMessage: {missing_parent_commit_message}")

        return {
            "total_bugs": total_bugs,
            "missing_fix_commit_message": missing_fix_commit_message,
            "missing_parent_commit_message": missing_parent_commit_message,
        }

    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return {
            "total_bugs": 0,
            "missing_fix_commit_message": 0,
            "missing_parent_commit_message": 0,
        }

def check_commit_messages_in_splits(base_path):
    """
    Check commit messages for train, test, and val files in both enhanced datasets
    """
    splits = ["sstubsLarge-train.json", "sstubsLarge-test.json", "sstubsLarge-val.json"]
    results = {}

    for split in splits:
        file_path = os.path.join(base_path, split)
        if os.path.exists(file_path):
            results[split] = check_commit_messages(file_path)
        else:
            print(f"File not found: {file_path}")

    print("\nSummary:")
    for split, result in results.items():
        print(f"{split}:")
        print(f"  Total bugs: {result['total_bugs']}")
        print(f"  Missing fixCommitMessage: {result['missing_fix_commit_message']}")
        print(f"  Missing parentCommitMessage: {result['missing_parent_commit_message']}\n")

if __name__ == "__main__":
    # Enhanced dataset directory for unique and repetition
    unique_enhanced_base_path = "../../Data/sstubs4j/unique/enhanced"
    repetition_enhanced_base_path = "../../Data/sstubs4j/repetition/enhanced"

    # Check commit messages in all splits
    check_commit_messages_in_splits(unique_enhanced_base_path)
    check_commit_messages_in_splits(repetition_enhanced_base_path)

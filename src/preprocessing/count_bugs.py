import json
import os
def count_bug_types(file_path):
    """
    Count occurrences of "bugType" in JSON file based from what type of file it is
    """
    try:
        print(f"Processing file: {file_path}")
        with open(file_path, 'r') as file:
            data = json.load(file)
            bug_type_count = sum(1 for item in data if "bugType" in item)
            print(f"Found {bug_type_count} 'bugType' entries in {file_path}")
            return bug_type_count
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return 0

def count_bug_types_in_splits(base_path, category):
    """
    Count "bugType" occurrences in train, test, and val files independently for unique and repititon
    """
    train_path = os.path.join(base_path, category, 'sstubsLarge-train.json')
    test_path = os.path.join(base_path, category, 'sstubsLarge-test.json')
    val_path = os.path.join(base_path, category, 'sstubsLarge-val.json')

    train_count = count_bug_types(train_path)
    test_count = count_bug_types(test_path)
    val_count = count_bug_types(val_path)

    # Display results
    print(f"\n{category.capitalize()} Dataset:")
    print(f"  Train 'bugType' count: {train_count}")
    print(f"  Test 'bugType' count: {test_count}")
    print(f"  Validation 'bugType' count: {val_count}")
    print(f"  Total 'bugType' count: {train_count + test_count + val_count}\n")

# Data folder where the directories exist locally (too m
base_path = '../Data'

# Process both repetition and unique datasets
count_bug_types_in_splits(base_path, 'repetition')
count_bug_types_in_splits(base_path, 'unique')


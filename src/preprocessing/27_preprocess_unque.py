import json
import os
import re

# -------- CONFIG --------
input_path = "../../Data/sstubs4j/unique/splits/sstubsLarge-test.json"
output_dir = "../../Data/27_sstubs_UNIQUE/"
os.makedirs(output_dir, exist_ok=True)

JAVA_KEYWORDS = set([
    "abstract", "assert", "boolean", "break", "byte", "case", "catch", "char",
    "class", "const", "continue", "default", "do", "double", "else", "enum",
    "extends", "final", "finally", "float", "for", "goto", "if", "implements",
    "import", "instanceof", "int", "interface", "long", "native", "new",
    "package", "private", "protected", "public", "return", "short", "static",
    "strictfp", "super", "switch", "synchronized", "this", "throw", "throws",
    "transient", "try", "void", "volatile", "while"
])

# -------- UTILS --------
def mask_identifiers(code):
    tokens = re.split(r'(\W)', code)
    id_map = {}
    id_counter = 1
    masked_tokens = []
    for tok in tokens:
        if re.match(r'^[a-zA-Z_]\w*$', tok) and tok not in JAVA_KEYWORDS:
            if tok not in id_map:
                id_map[tok] = f"VAR{id_counter}"
                id_counter += 1
            masked_tokens.append(id_map[tok])
        else:
            masked_tokens.append(tok)
    return ''.join(masked_tokens)

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

# -------- CORE --------
def create_examples(dataset):
    processed = []
    for item in dataset:
        context = item.get("contextBefore", "").strip()
        fix_commit = item.get("fixCommitMessage", "").strip()
        parent_commit = item.get("parentCommitMessage", "").strip()

        buggy = item.get("buggyCode", "").strip()
        before = item.get("sourceBeforeFix", "").strip()
        after = item.get("sourceAfterFix", "").strip()

        if not buggy or not before or not after or not context:
            continue

        # Aligned fix reconstruction
        if before in buggy:
            fixed = buggy.replace(before, after)
        else:
            fixed = after

        #  Mask
        context = mask_identifiers(context)
        buggy = mask_identifiers(buggy)
        fixed = mask_identifiers(fixed)

        # ➕ Buggy (positive)
        buggy_input = f"[CONTEXT] {context}\n[SNIPPET] {buggy}\n[COMMIT] {fix_commit}\n[PARENT] {parent_commit}"
        processed.append({"text": buggy_input, "label": 1})

        # ➖ Fixed (negative)
        fixed_input = f"[CONTEXT] {context}\n[SNIPPET] {fixed}\n[COMMIT] {fix_commit}\n[PARENT] {parent_commit}"
        processed.append({"text": fixed_input, "label": 0})

    return processed

# -------- RUN --------
print(f"Processing unique test set from: {input_path}")
data = load_json(input_path)
processed = create_examples(data)
output_path = os.path.join(output_dir, "test.json")

with open(output_path, "w") as f:
    json.dump(processed, f, indent=2)

print(f" Saved processed unique test set to: {output_path}")

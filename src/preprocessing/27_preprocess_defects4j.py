import os
import json
import re

# -------- CONFIG --------
input_path = "../../Data/defects4j/splits/defects4j-test.json"
output_path = "../../Data/27_DEFECTS4J/test.json"
JAVA_KEYWORDS = set([
    "abstract", "assert", "boolean", "break", "byte", "case", "catch", "char",
    "class", "const", "continue", "default", "do", "double", "else", "enum",
    "extends", "final", "finally", "float", "for", "goto", "if", "implements",
    "import", "instanceof", "int", "interface", "long", "native", "new",
    "package", "private", "protected", "public", "return", "short", "static",
    "strictfp", "super", "switch", "synchronized", "this", "throw", "throws",
    "transient", "try", "void", "volatile", "while"
])

# -------- MASKING UTILITY --------
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

# -------- SNIPPET EXTRACTOR --------
def extract_snippet(diff_text):
    lines = diff_text.split("\n")
    snippet_lines = []
    for line in lines:
        if line.startswith("+") and not line.startswith("+++"):
            snippet_lines.append(line[1:].strip())
    return "\n".join(snippet_lines).strip()

# -------- LOAD DATA --------
with open(input_path, "r") as f:
    data = json.load(f)

processed = []

# -------- MAIN LOOP --------
for entry in data:
    diff = entry.get("diff", "")
    snippet = extract_snippet(diff)
    if not snippet:
        continue

    # Handle changedFiles as dict or list
    changed_files = entry.get("changedFiles", [])
    if isinstance(changed_files, dict):
        files = list(changed_files.keys())
    elif isinstance(changed_files, list):
        files = changed_files
    else:
        files = []

    context = files[0] if files else "Unknown.java"
    project = entry.get("program", "unknown")

    #  Apply masking
    context = mask_identifiers(context)
    snippet = mask_identifiers(snippet)

    # Format into model input
    text = f"[CONTEXT] {context}\n[SNIPPET] {snippet}\n[COMMIT] Fix in project {project}\n[PARENT] Pre-fix version"
    processed.append({
        "text": text,
        "label": 1  # Assume all are buggy
    })

# -------- SAVE --------
with open(output_path, "w") as f:
    json.dump(processed, f, indent=2)

print(f" Saved {len(processed)} processed buggy examples to: {output_path}")

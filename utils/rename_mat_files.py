import os
import re

DATASET_DIR = "dataset/BCICIV2a"

# Regex para arquivos tipo A01T.mat → parsed_P01T.mat
pattern = re.compile(r"A(\d{2})([TE])\.mat", re.IGNORECASE)

for filename in os.listdir(DATASET_DIR):
    if filename.lower().endswith(".mat"):
        match = pattern.match(filename)
        if match:
            subject, session = match.groups()
            new_name = f"parsed_P{subject}{session}.mat"
            old_path = os.path.join(DATASET_DIR, filename)
            new_path = os.path.join(DATASET_DIR, new_name)
            os.rename(old_path, new_path)
            print(f"✔️ {filename} → {new_name}")

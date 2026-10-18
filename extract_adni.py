import zipfile
import os
from pathlib import Path

extract_dir = Path("data/raw/ADNI")
extract_dir.mkdir(parents=True, exist_ok=True)

zips = [
    "C:\\Users\\navya\\Downloads\\ADNI_T1.zip",
    "C:\\Users\\navya\\Downloads\\ADNI_T11.zip",
    "C:\\Users\\navya\\Downloads\\ADNI_T1_dataset.zip",
]

for zip_file in zips:
    if os.path.exists(zip_file):
        print(f"Extracting {Path(zip_file).name}...")
        with zipfile.ZipFile(zip_file, 'r') as z:
            z.extractall(extract_dir)
        print(f"✓ Done: {Path(zip_file).name}")

print("\n✓ All extractions complete!")
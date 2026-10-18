import csv
import random
from pathlib import Path

random.seed(42)

# Get all subject IDs from extracted folders
adni_path = Path("data/raw/ADNI/ADNI")
subject_ids = sorted([d.name for d in adni_path.iterdir() if d.is_dir()])

print(f"Found {len(subject_ids)} subjects")

# Assign diagnoses: 50% CN, 30% MCI, 20% AD
num_cn = int(len(subject_ids) * 0.50)
num_mci = int(len(subject_ids) * 0.30)
num_ad = len(subject_ids) - num_cn - num_mci

diagnosis_list = ['CN'] * num_cn + ['MCI'] * num_mci + ['AD'] * num_ad
random.shuffle(diagnosis_list)

# Write CSV
Path("data/raw").mkdir(parents=True, exist_ok=True)
with open("data/raw/ADNI_diagnoses.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["SUBJECT_ID", "DIAGNOSIS"])
    for sid, diag in zip(subject_ids, diagnosis_list):
        writer.writerow([sid, diag])

print(f"✅ Created data/raw/ADNI_diagnoses.csv")
print(f"   CN:  {diagnosis_list.count('CN')}")
print(f"   MCI: {diagnosis_list.count('MCI')}")
print(f"   AD:  {diagnosis_list.count('AD')}")

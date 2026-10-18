"""
Build full ADNI diagnosis labels from the IDA collection CSV, cross-checked
against the metadata XML files.

Reads:   data/raw/ADNI_collection.csv      (IDA image list: Subject, Group, ...)
         data/raw/ADNI_metadata/**/*.xml   (subjectIdentifier + researchGroup)
         data/raw/ADNI/ADNI/<SUBJECT_ID>/  (image folders on disk)
Writes:  data/raw/ADNI_diagnoses_full.csv  (NEW file; refuses to overwrite)

Does not modify or delete anything else.

Group mapping (research decision, edit here if needed):
  CN -> CN,   MCI / EMCI / LMCI -> MCI,   AD -> AD,   SMC -> excluded
"""

import csv
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
COLLECTION_CSV = ROOT / "data" / "raw" / "ADNI_collection.csv"
META_DIR = ROOT / "data" / "raw" / "ADNI_metadata"
IMAGE_DIR = ROOT / "data" / "raw" / "ADNI" / "ADNI"
OUT_FILE = ROOT / "data" / "raw" / "ADNI_diagnoses_full.csv"

GROUP_MAP = {"CN": "CN", "MCI": "MCI", "EMCI": "MCI", "LMCI": "MCI", "AD": "AD"}
EXCLUDED = {"SMC"}


def fmt(counter):
    return ", ".join(f"{k}={v}" for k, v in sorted(counter.items()))


def read_collection():
    groups = defaultdict(Counter)
    n_rows = 0
    with open(COLLECTION_CSV, newline="", encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            sid = (row.get("Subject") or "").strip()
            group = (row.get("Group") or "").strip().upper()
            if sid and group:
                groups[sid][group] += 1
                n_rows += 1
    return groups, n_rows


def read_xml_groups():
    groups = defaultdict(Counter)
    for f in META_DIR.rglob("*.xml"):
        try:
            root = ET.parse(f).getroot()
        except ET.ParseError:
            continue
        sid = group = None
        for el in root.iter():
            tag = el.tag.split("}")[-1]
            if tag == "subjectIdentifier" and sid is None:
                sid = (el.text or "").strip()
            elif tag == "researchGroup" and group is None:
                group = (el.text or "").strip().upper()
            if sid and group:
                break
        if sid and group:
            groups[sid][group] += 1
    return groups


def main():
    if OUT_FILE.exists():
        sys.exit(f"[Labels] {OUT_FILE} already exists - not overwriting. Rename or move it first.")
    if not COLLECTION_CSV.exists():
        sys.exit(f"[Labels] {COLLECTION_CSV} not found.")

    csv_groups, n_rows = read_collection()
    xml_groups = read_xml_groups() if META_DIR.exists() else {}
    print(f"[Labels] Collection CSV: {n_rows} image rows, {len(csv_groups)} subjects")
    print(f"[Labels] Metadata XML:   {len(xml_groups)} subjects with a research group")

    # 1. Subjects whose rows in the CSV disagree with each other
    inconsistent = {s: c for s, c in csv_groups.items()
                    if len({GROUP_MAP.get(g, g) for g in c}) > 1}

    # 2. Cross-check CSV vs XML for subjects present in both
    both = sorted(set(csv_groups) & set(xml_groups))
    disagree = []
    for s in both:
        csv_g = GROUP_MAP.get(csv_groups[s].most_common(1)[0][0], csv_groups[s].most_common(1)[0][0])
        xml_g = GROUP_MAP.get(xml_groups[s].most_common(1)[0][0], xml_groups[s].most_common(1)[0][0])
        if csv_g != xml_g:
            disagree.append((s, dict(csv_groups[s]), dict(xml_groups[s])))
    print(f"\n[Check] Subjects in both CSV and XML: {len(both)}")
    print(f"[Check]   same group in both:  {len(both) - len(disagree)}")
    print(f"[Check]   DIFFERENT group:     {len(disagree)}")
    for s, c, x in disagree[:10]:
        print(f"            {s}: CSV {c}  vs  XML {x}")
    print(f"[Check] Subjects with mixed groups inside the CSV: {len(inconsistent)}")
    for s, c in list(inconsistent.items())[:10]:
        print(f"            {s}: {dict(c)}")
    xml_only = sorted(set(xml_groups) - set(csv_groups))
    if xml_only:
        print(f"[Check] Subjects only in XML (not in CSV): {len(xml_only)}, e.g. {xml_only[:5]}")

    # 3. Build labels for subjects that have an image folder on disk
    folders = {p.name for p in IMAGE_DIR.iterdir() if p.is_dir()} if IMAGE_DIR.exists() else set()
    rows, raw_counts, final_counts = [], Counter(), Counter()
    skipped = Counter()
    for sid in sorted(folders):
        if sid in csv_groups:
            source, counter = "collection_csv", csv_groups[sid]
        elif sid in xml_groups:
            source, counter = "metadata_xml", xml_groups[sid]
        else:
            skipped["no label in CSV or XML"] += 1
            continue
        if sid in inconsistent or any(d[0] == sid for d in disagree):
            skipped["conflicting groups"] += 1
            continue
        raw = counter.most_common(1)[0][0]
        raw_counts[raw] += 1
        if raw in EXCLUDED or raw not in GROUP_MAP:
            skipped[f"group {raw} excluded"] += 1
            continue
        diagnosis = GROUP_MAP[raw]
        final_counts[diagnosis] += 1
        rows.append({"SUBJECT_ID": sid, "DIAGNOSIS": diagnosis,
                     "RESEARCH_GROUP": raw, "SOURCE": source})

    print(f"\n[Labels] Image folders on disk: {len(folders)}")
    print(f"[Labels] Original ADNI groups (folders with a label): {fmt(raw_counts)}")
    print(f"[Labels] Skipped: {fmt(skipped) if skipped else 'none'}")
    print("[Labels] FINAL: " + ", ".join(f"{d}={final_counts[d]}" for d in ("CN", "MCI", "AD"))
          + f"  (total {len(rows)} of {len(folders)} folders)")

    with open(OUT_FILE, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["SUBJECT_ID", "DIAGNOSIS", "RESEARCH_GROUP", "SOURCE"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n[Labels] Wrote {len(rows)} rows to {OUT_FILE}")


if __name__ == "__main__":
    main()

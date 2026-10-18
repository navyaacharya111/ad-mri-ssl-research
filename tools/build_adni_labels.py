"""
Build real ADNI diagnosis labels from the IDA metadata XML files.

Reads:   data/raw/ADNI_metadata/**/*.xml   (subjectIdentifier + researchGroup)
         data/raw/ADNI/ADNI/<SUBJECT_ID>/  (to check which image folders get a label)
Writes:  data/raw/ADNI_diagnoses_real.csv  (NEW file; refuses to overwrite)

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
META_DIR = ROOT / "data" / "raw" / "ADNI_metadata"
IMAGE_DIR = ROOT / "data" / "raw" / "ADNI" / "ADNI"
OUT_FILE = ROOT / "data" / "raw" / "ADNI_diagnoses_real.csv"

GROUP_MAP = {"CN": "CN", "MCI": "MCI", "EMCI": "MCI", "LMCI": "MCI", "AD": "AD"}
EXCLUDED = {"SMC"}


def read_xml(path):
    """Return (subject_id, research_group) or (None, None) if the file has no subject block."""
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError:
        return None, None
    sid = group = None
    for el in root.iter():
        tag = el.tag.split("}")[-1]
        if tag == "subjectIdentifier" and sid is None:
            sid = (el.text or "").strip()
        elif tag == "researchGroup" and group is None:
            group = (el.text or "").strip().upper()
        if sid and group:
            break
    return sid or None, group or None


def main():
    if OUT_FILE.exists():
        sys.exit(f"[Labels] {OUT_FILE} already exists - not overwriting. Rename or move it first.")

    xml_files = sorted(META_DIR.rglob("*.xml"))
    print(f"[Labels] XML files found: {len(xml_files)}")

    groups_by_subject = defaultdict(Counter)
    no_group = []
    for f in xml_files:
        sid, group = read_xml(f)
        if sid and group:
            groups_by_subject[sid][group] += 1
        else:
            no_group.append(f.name)

    print(f"[Labels] XMLs with subject + researchGroup: {len(xml_files) - len(no_group)}")
    print(f"[Labels] XMLs without a researchGroup:      {len(no_group)}")
    for name in no_group[:3]:
        print(f"           e.g. {name}")

    # One label per subject; flag subjects whose XMLs disagree
    rows, conflicts, excluded = [], [], []
    raw_counts, final_counts = Counter(), Counter()
    for sid in sorted(groups_by_subject):
        counter = groups_by_subject[sid]
        raw_group = counter.most_common(1)[0][0]
        raw_counts[raw_group] += 1
        mapped = {GROUP_MAP.get(g, g) for g in counter}
        if len(mapped) > 1:
            conflicts.append((sid, dict(counter)))
            continue
        if raw_group in EXCLUDED or raw_group not in GROUP_MAP:
            excluded.append((sid, raw_group))
            continue
        diagnosis = GROUP_MAP[raw_group]
        final_counts[diagnosis] += 1
        rows.append({"SUBJECT_ID": sid, "DIAGNOSIS": diagnosis,
                     "RESEARCH_GROUP": raw_group, "N_XML": sum(counter.values())})

    print(f"\n[Labels] Subjects with metadata: {len(groups_by_subject)}")
    print("[Labels] Original ADNI groups (per subject): " +
          ", ".join(f"{g}={n}" for g, n in sorted(raw_counts.items())))
    print(f"[Labels] Excluded (SMC / unknown group): {len(excluded)}")
    print(f"[Labels] Conflicting groups across a subject's XMLs (left out): {len(conflicts)}")
    for sid, c in conflicts[:10]:
        print(f"           {sid}: {c}")
    print("[Labels] Final labels: " +
          ", ".join(f"{d}={final_counts[d]}" for d in ("CN", "MCI", "AD")) +
          f"  (total {len(rows)})")

    # Coverage against the image folders actually on disk
    if IMAGE_DIR.exists():
        folders = {p.name for p in IMAGE_DIR.iterdir() if p.is_dir()}
        labelled = {r["SUBJECT_ID"] for r in rows}
        print(f"\n[Labels] Image folders on disk: {len(folders)}")
        print(f"[Labels]   with a final label:   {len(folders & labelled)}")
        print(f"[Labels]   without a label:      {len(folders - labelled)}")
        missing = sorted(folders - set(groups_by_subject))
        if missing:
            print(f"[Labels]   no XML at all ({len(missing)}), e.g. {missing[:5]}")
        extra = sorted(labelled - folders)
        if extra:
            print(f"[Labels] Labels with no image folder (ignored by training): {len(extra)}")

    with open(OUT_FILE, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["SUBJECT_ID", "DIAGNOSIS", "RESEARCH_GROUP", "N_XML"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n[Labels] Wrote {len(rows)} rows to {OUT_FILE}")


if __name__ == "__main__":
    main()

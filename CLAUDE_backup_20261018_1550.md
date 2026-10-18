# AD-MRI SSL Research Project — context for Claude Code

This file gives you (Claude Code) the full history, current state and rules of this project.
Read it fully before doing anything. The owner is an MCA student; explain things in plain
language and give step-by-step instructions when asking them to do something.

---

## 1. Research question

**"Does self-supervised contrastive pretraining (SimCLR) improve cross-dataset generalization
of Alzheimer's disease MRI classification compared with conventional supervised training?"**

- Task: 3-class classification of structural T1 MRI — CN (cognitively normal), MCI (mild
  cognitive impairment), AD (Alzheimer's disease).
- Supervised arm: ResNet-18 trained from scratch on labels.
- SSL arm: SimCLR contrastive pretraining of the same ResNet-18 encoder (no labels), then
  (a) full fine-tuning and (b, from v3) a linear probe on the frozen encoder.
- Source dataset: ADNI. Target dataset for cross-cohort testing: OASIS — **NOT obtained yet,
  so cross-dataset generalization has NOT been tested at all.** All results so far are within-ADNI.
- Metrics: subject-level accuracy, macro precision/recall/F1 (also ROC-AUC planned), and the
  generalization gap (within-ADNI minus cross-OASIS performance) once OASIS exists.
- Leakage prevention: subject-level train/val/test split (70/15/15), slices of one subject never
  cross splits, predictions aggregated per subject (mean slice probability -> argmax).

---

## 2. Environment

- Windows laptop, **PowerShell**, CPU only (no GPU). Project folder:
  `C:\Users\navya\ad-mri-ssl-research`
- Virtual environment: `.\venv` — activate with `.\venv\Scripts\Activate.ps1` (prompt shows `(venv)`).
  If a new window says a module is missing, the venv is not active.
- Key packages: torch/torchvision, albumentations **2.0.8**, pydicom, scipy, pandas, numpy, tqdm.
- Set `$env:NO_ALBUMENTATIONS_UPDATE = "1"` before runs (silences a harmless online version check).
- A full experiment takes **1.5–2 days** on this CPU. Before any long run the user must pause
  Windows Update (a Windows Update restart killed a 2-day run on 12 Oct 2026) and keep the
  charger plugged in with sleep set to Never.

---

## 3. Project layout

```
config.py                     settings (paths, MRI_SIZE=128, RANDOM_SEED=42, splits, epochs, SSL_CONFIG)
src/dataset.py                subject index + DICOM loading + slice selection (v2, see §5)
src/splitter.py               subject-level splits + leakage check (original, unchanged)
src/dataloader.py             PyTorch slice datasets/loaders, albumentations (fixed, see §6)
src/models.py                 ResNet18Classifier, ResNet18Encoder, SimCLR projection head (unchanged)
src/train.py                  SupervisedTrainer, SimCLRTrainer, NT-Xent, checkpointing (rewritten)
src/evaluate.py               Evaluator.evaluate_subject_level, compare_models (original, unchanged)
src/experiment.py             full pipeline, crash-safe, EXPERIMENT_VERSION tag (rewritten)
tools/build_adni_labels.py        labels from metadata XMLs only  -> ADNI_diagnoses_real.csv
tools/build_adni_labels_full.py   labels from IDA collection CSV + XML cross-check -> ADNI_diagnoses_full.csv
tools/update_config.py            one-off config edit (pretrain_epochs, temperature)
tools/check_slices.py             saves a PNG of exactly which slices the model sees for a subject
apply_fixes.py / apply_fix2.py / apply_fix3.py   installers used to apply the fixes below
backup_before_fixes_*/ backup_before_fix2_*/ backup_before_fix3_*/   original files before each fix
config_backup_*.py            config.py backups
outputs/results/runs/         one JSON per finished run (saved immediately after each run)
outputs/results/              final experiment_results_*.json and statistics_*.json (date-stamped)
outputs/checkpoints/<run tag>/   per-stage checkpoints (supervised, simclr_pretrain, ssl_linear_probe, ssl_finetune)
outputs/logs/                 full console log of every run
outputs/plots/                slice_check_*.png images
```

---

## 4. Data and labels (important provenance)

- Images: `data/raw/ADNI/ADNI/<SUBJECT_ID>/<SERIES>/<DATE>/<IMAGE_ID>/*.dcm` — 398 subject folders,
  T1 MPRAGE, **acquired SAGITTALLY**, ~1.0–1.2 mm slices, ~170–208 slices per scan. Some subjects
  have several series (e.g. MPRAGE + MPRAGE_REPEAT, localizers); dataset.py picks the MPRAGE-type
  folder with the most slices (≥ 40).
- Metadata: `data/raw/ADNI_metadata/ADNI/*.xml` — 2,320 XMLs; 459 contain `<researchGroup>`
  (237 subjects). The others are series/image-level metadata without diagnosis.
- `data/raw/ADNI_collection.csv` — IDA image-collection export downloaded 14 Oct 2026
  (1,861 image rows, 1,060 subjects; columns `Subject`, `Group`, ...).
- **Labels used for training:** `data/raw/ADNI_diagnoses.csv` = copy of `ADNI_diagnoses_full.csv`
  (388 subjects: CN 158, MCI 194 [= MCI 151 + EMCI 28 + LMCI 15], AD 36; 10 SMC excluded).
  CSV and XML agreed on 237/237 overlapping subjects.
- 387 subjects load (1 MCI subject has no usable 3D scan).
- `data/raw/ADNI_diagnoses_PLACEHOLDER_random.csv` = **fake random labels** created early in the
  project (CN 199 / MCI 119 / AD 80, seed 42). Never use it for results.
- Group mapping decision: EMCI/LMCI -> MCI, SMC excluded.

---

## 5. Pipeline as it is now

- **Slice selection (v2, `dataset.py`):** slice axis resampled to 1 mm using DICOM positions /
  SliceThickness; head midline found automatically (centre of mass); **16 sagittal slices per
  hemisphere, 20–38 mm from the midline** (hippocampal region) = 32 slices/subject, resized to
  128×128, intensity normalized by 1st–99th percentile to [0,1], replicated to 3 channels.
- Train split ≈ 270 subjects (8,640 slices), val ≈ 56, test ≈ 61 (only ~6 AD in test).
- Class-weighted cross-entropy (inverse frequency from training subjects, ≈ CN 0.82, MCI 0.67, AD 3.6).
- **SimCLR (`train.py`):** two independent augmented views per image (`simclr_augment`: random
  resized crop 0.6–1.0, flip, ±15° rotation, brightness/contrast, blur, light noise) and standard
  NT-Xent loss (2N−2 negatives), temperature 0.5, batch 16, projection dim 128.
- **Crash safety:** atomic checkpoint after every epoch; restart resumes the last finished epoch;
  finished stages and finished runs are skipped; final results get date-stamped filenames;
  `EXPERIMENT_VERSION` is part of every run/checkpoint name so a method change never reuses old runs.
- 2 runs, seeds 43 and 44 (`ExperimentRunner(num_runs=2, ...)` at the bottom of experiment.py).
- Run command: `python src/experiment.py` (re-run the same command to resume after an interruption).

---

## 6. Bugs found and fixed (keep these fixed)

1. Early `dataset.py` returned **random noise** instead of MRI slices — fixed (real DICOM loading).
2. Labels were **fake random placeholders** — replaced by real IDA/XML labels (§4).
3. Original SimCLR "second view" was the same image + 0.01 noise (not SimCLR) — replaced with
   real two-view augmentation + standard NT-Xent.
4. `A.Normalize(mean=0.5, std=0.5)` on 0–1 images squeezed inputs to [−1.00, −0.99] — fixed with
   `max_pixel_value=1.0`. Default `GaussNoise` was far too strong — now `std_range=(0.01, 0.03)`.
5. Early stopping kept the last epoch instead of the best — fixed (best weights restored).
6. No checkpoints/incremental saving — fixed (§5).
7. Middle 32 sagittal slices covered only ±16 mm around the midline and missed the hippocampus —
   fixed by the v2 hippocampal slice selection (verified visually with tools/check_slices.py).
8. SimCLR temperature changed 0.07 -> 0.5 (small batch); pretraining epochs 100 -> 50 (v2) -> 30 (v3).

---

## 7. Results so far (do not invent or alter these numbers)

| Version | Description | Supervised acc / macro-F1 | SSL fine-tuned acc / macro-F1 | Notes |
|---|---|---|---|---|
| mock test (8 Oct) | fake data | 33.3% | — | `outputs/results/experiment_results.json` (+ backup folder `results_backup_2026-10-08`) |
| run 10–12 Oct | placeholder labels | — | — | killed by Windows Update, nothing saved |
| v1 (`adni`) | real labels, middle 32 sagittal slices | 42.6% / 0.304 | 36.9% / 0.312 | near chance; all trainings early-stopped at epoch 11 (best = epoch 1) |
| v2 (`v2_hippocampal`) | hippocampal slices | 44.3% / 0.315 (recall 0.341) | 35.3% / 0.259 (recall 0.271) | p = 0.27–0.63 (2 runs), not significant |

v2 details: `outputs/results/statistics_adni_v2_hippocampal_20261017_115724.json`. Supervised train
accuracy reached ~97% while validation stayed ~45–55% (severe overfitting). Model selection by
validation loss picked epoch 1–2; for SSL fine-tuning that meant an almost untrained model
(train acc ~0.42). SimCLR loss fell 2.69 -> 1.64 (pretraining itself works).

**Honest conclusion so far:** with 2D slices, no skull-stripping/registration and ~270 training
subjects, both approaches are near chance on 3-class ADNI and there is no significant difference.
Nothing can yet be said about SSL improving (or harming) cross-dataset generalization.

---

## 8. v3 (prepared 17 Oct 2026 — check whether it is installed / running / finished)

`apply_fix3.py` installs: best epoch chosen by **subject-level validation macro-F1** (not loss);
AdamW weight decay 0.01; label smoothing 0.1; supervised lr 3e-4; SimCLR 30 epochs; new
**linear-probe arm** (`FrozenEncoderClassifier`, frozen encoder copy + linear layer); statistics for
both SSL arms vs supervised. Version tag `v3_f1select_reg_probe`; settings in `V3_SETTINGS` in
experiment.py.

Check state with:
`Select-String -Path src\experiment.py -Pattern "EXPERIMENT_VERSION ="` and
`Get-ChildItem outputs\results\runs, outputs\results\statistics_* | Sort-Object LastWriteTime`.

---

## 9. Next steps (in order)

1. Find out whether v3 is installed, running, or finished; if finished, analyse
   `statistics_adni_v3_*.json` and the run JSONs (confusion matrices, `history_*`, best epochs).
2. If performance is still near chance, discuss options with the user before any new 2-day run:
   e.g. CN-vs-AD binary task as a sanity check, more runs (5 seeds) for statistics, coronal
   hippocampal slices, skull-stripping/registration, freezing early layers.
3. **OASIS:** the user must request/download it (OASIS-1 or OASIS-3 from oasis-brains.org; check
   access terms). Then: write a loader for OASIS with the SAME preprocessing and slice selection,
   map labels (e.g. CDR 0 = CN, 0.5 = MCI/very mild, ≥1 = AD — agree the mapping with the user),
   evaluate the already-trained ADNI models on OASIS **without training on OASIS**, and compute the
   generalization gap per arm.
4. Write up results for the MCA report/paper.

---

## 10. Rules — always follow

- **Never delete or overwrite** data, labels, results, checkpoints, logs or backups. Back up a file
  before editing it (`backup_before_<change>_<timestamp>/`). New result files must get new names.
- **Bump `EXPERIMENT_VERSION`** in experiment.py whenever the method changes, so resume never
  mixes runs from different methods.
- **Do not start long training runs yourself** without the user's explicit OK; state the expected
  duration and remind them to pause Windows Update first.
- **Test every code change** on mock/synthetic data (and a simulated interruption + resume for
  pipeline changes) before the user runs it on real data.
- Use **PowerShell** syntax for commands. Explain what each command does. Prefer read-only checks
  first. When quoting code from a file (not to be run), label it as a quote.
- **Research honesty:** report only numbers that exist in the result files; never claim SSL is
  better/worse without statistical support; with only 2 runs the paired t-test has almost no power —
  say so; clearly separate completed work from planned work; cross-dataset claims require OASIS results.
- **References:** use only real, verifiable papers with correct authors/venues/DOIs. The reference
  lists in the earlier Word documents (`AD_MRI_SSL_Research_Report.docx`,
  `AD_MRI_SSL_Research_Proposal.docx`) were written before any results and were NOT verified —
  some entries are wrong or irrelevant (e.g. a "Morris et al. 2001 Unified Parkinson's Disease
  Rating Scale" entry, a mis-titled Cheplygina et al. entry, unrelated segmentation papers).
  Re-check every reference before reuse. Those documents also describe results/plans that are now
  outdated (e.g. "398 subjects", placeholder-label era) — do not copy numbers from them.

# ECG Arrhythmia Classifier

Beat-level arrhythmia classification on the MIT-BIH Arrhythmia Database, built to the ANSI/AAMI EC57 standard — with an emphasis on **verifying the pipeline against ground truth at every stage** rather than reporting a headline accuracy number.

> **Project status: active development.** The detector and classifier both run end-to-end across all 48 records. Two defects have been identified, quantified, and are pending fixes (see [Known Defects](#known-defects)). Classification metrics below are reported as a **provisional baseline** and are expected to change once those fixes land. They should not be read as final performance.

---

## Why this repo is structured around validation

Published results on MIT-BIH vary enormously depending on evaluation protocol. Intra-patient splits — where beats from the same subject appear in both training and test sets — routinely report >98% accuracy and are not clinically meaningful, because a deployed device never sees the patient it was trained on. This project uses the **inter-patient paradigm** throughout.

More importantly: the classifier consumes the output of a QRS detector. If the detector is wrong, every downstream feature and metric is built on corrupted input, and no amount of model tuning will surface that. A substantial portion of the work here is measuring detector error before trusting classifier results.

---

## Dataset

[MIT-BIH Arrhythmia Database](https://physionet.org/content/mitdb/1.0.0/) — 48 half-hour two-lead recordings at 360 Hz with expert beat-by-beat annotations. Accessed via `wfdb`. Data is not tracked in this repo.

```python
import wfdb
wfdb.dl_database('mitdb', dl_dir='data/mitdb')
```

### Annotation handling

Raw WFDB annotation files mix genuine beat labels with non-beat markers (rhythm changes, signal-quality flags, comments, flutter-wave boundaries). Counting these as beats inflates apparent detector miss rates.

An `AAMI_MAP` dictionary was built mapping the 15 genuine beat symbols (`N L R e j A a J S V E F / f Q`) to the five AAMI classes; every other symbol is filtered before any count comparison.

**Validation:** after filtering, the summed beat count across all 48 records is **109,494 — matching the published MIT-BIH total exactly.** The stripped non-beat annotations total **3,153**, independently matching the discrepancy observed before the correction. Two independently-derived figures reconciling to the same numbers is what justified trusting the downstream analysis.

This correction materially changed the picture. Record 231 contained 427 comment annotations (`"`), each counted as a missed beat — its apparent miss rate fell from 21.9% to **0.1%** once filtered. Records 219, 200, and 222 showed similar collapses. Most of the "detector problem" was an annotation-parsing problem.

---

## Pipeline

### Preprocessing

| Stage | Implementation | Rationale |
|---|---|---|
| Baseline wander removal | 2nd-order Butterworth high-pass @ 0.5 Hz, `filtfilt` | AHA-recommended corner. Zero-phase filtering prevents the phase distortion that would shift R-peak positions and invalidate timing-tolerance scoring. |
| Powerline interference | Dual notch @ 60 Hz **and** 120 Hz | The 2nd harmonic is present in MIT-BIH records; most implementations remove only the fundamental. |
| High-frequency noise | 2nd-order Butterworth low-pass @ 40 Hz | |
| Normalization | Per-record z-score | Global normalization computed across all records would leak test-set statistics into training. |

### R-peak detection — Pan-Tompkins, implemented from scratch

1. 5–15 Hz bandpass isolating QRS spectral content
2. **5-point Lagrange differentiation stencil `[-1,-8,0,8,1]/12h`**
3. Squaring (rectification + emphasis of high slopes)
4. 150 ms moving-window integration
5. Adaptive dual thresholding
6. Peak refinement: `argmax` in the cleaned signal within ±50 ms of the integrator-detected peak

**On the derivative stencil:** the original 1985 paper specifies `[-1,-2,0,2,1]/8h`. That is a lower-order approximation chosen for real-time integer arithmetic on 1980s hardware. The `[-1,-8,0,8,1]/12h` stencil used here was derived from the full 5-point Lagrange interpolating polynomial and is the more accurate estimator; the derivation was worked through symbolically rather than taken on faith.

### Feature extraction

- **RR interval features (4):** current RR, pre/post ratio, local mean, normalized RR
- **Wavelet features (18):** db4, 5-level decomposition — energy, mean, std per sub-band
- **P-wave features (8):** signed peak amplitude, band energy, db2 level-3 approximation coefficients
- **Current feature matrix: 30 features × 98,384 beats**

The P-wave block is presently included in the live pipeline but is **flagged for removal** — it did not improve results and its extraction window was never validated. See [P-wave feature investigation](#p-wave-feature-investigation).

### Classification

- Random Forest, `class_weight='balanced'`, `random_state=42`
- Patient-wise holdout split (38 train / 10 test)
- 5-fold `GroupKFold` cross-validation grouped on patient ID
- SMOTE oversampling applied **inside training folds only**, targeting 10% of the N-class count for S and V, with a guard skipping resampling when the target is already met

---

## Detector validation (ANSI/AAMI EC57)

A beat-matching harness was built to score detected R-peaks against reference annotations:

- **Two-pointer matching** with a **±150 ms (±54 sample) tolerance window**, per EC57
- Reference beats filtered through `AAMI_MAP` before matching
- Explicit cleanup sweep for the leftover pointer after the main loop
- **Verified against a hand-traced synthetic case before being run on real data**
- Guarded by two balance assertions: `matches + misses == n_true` and `matches + false_alarms == n_detected`
- Returns raw `(matches, misses, false_alarms)` counts per record — deliberately **not** percentages, since gross sensitivity is not recoverable from a list of per-record rates

Sensitivity = `matches / (matches + misses)`; positive predictivity = `matches / (matches + false_alarms)`.

### Results: six records account for essentially all detection error

| Record | True beats | Detected | Error | Failure mode |
|---|---|---|---|---|
| **228** | 2,053 | 358 | **82.6% missed** | Adaptive threshold fails to recover after an amplitude excursion — *working hypothesis, partially verified* |
| **104** | 2,229 | 922 | **58.6% missed** | Paced record; miss mechanism plausible but **not verified to the same standard as 228** |
| **114** | 1,879 | 1,234 | **34.3% missed** | Unexplained. Lead reversal is documented for this record in PhysioNet — *hypothesis, untested* |
| **207** | 1,860 | 2,053 | **10.4% over-detected** | Ventricular flutter — continuous sinusoidal waveform with no discrete per-cycle QRS causes multiple firings per cycle. Morphology problem, not a threshold problem. |
| **108** | 1,763 | 1,929 | **9.4% over-detected** | Unexplained |
| 203 | 2,980 | 2,774 | 6.9% missed | Literature-documented noisy record; not yet investigated |

**All 42 remaining records fall under 2% error, most under 0.5%.**

### Bin-density diagnostic

Aggregate Se/+P tells you *how badly* a record fails but not *how*. A windowed diagnostic computing sensitivity per 30-second bin (10,800 samples) across each full record was built to characterize the **temporal structure** of failure.

This distinguishes failure classes that aggregate metrics conflate: record 228 shows discrete collapse-and-recovery episodes consistent with threshold staleness, whereas record 207 shows uniformly distributed over-detection consistent with a morphology mismatch. The two require entirely different fixes; the aggregate number alone would not have revealed that.

An investigation into record 228's first collapse window found elevated V-beat density (6.0% vs 2.3% baseline), but the second collapse window contained **zero** V beats — refuting a single PVC-cluster explanation. A follow-on check refuted an initial "lower amplitude" hypothesis: the bad window's median integrated peak height was *higher* than a comparable good window. This analysis is noted as incomplete — `find_peaks` was run without a `prominence` constraint and is picking up 3–4× more local maxima than real beats, which dilutes the medians. The directional finding is probably sound; it has not been cleaned up.

---

## Known defects

### 1. Beat segmentation window contamination

`segment_beats()` currently extracts a fixed 360-sample window (180 samples either side of the R-peak). For beats with short RR intervals, this window **overlaps the neighbouring beat**, injecting foreign QRS energy into the sample.

Quantified by constructing leading- and trailing-edge boolean masks (exclusive-end slice semantics give true thresholds of 180/360 s and 179/360 s respectively):

| Class | Beats affected |
|---|---|
| V (ventricular) | **~54%** |
| F (fusion) | **~36%** |
| S (supraventricular) | **~30%** |
| N (normal) | ~6.5% |
| **Overall** | **10,583 / 98,384 (~10.75%)** |

The contamination is concentrated precisely in the minority classes the model exists to detect.

**Mechanism confirmed empirically, not assumed.** db4 D1/D2 sub-band energy in contaminated V beats is **2.3–2.7× higher** than in clean V beats. Critically, *near-threshold-but-clean* beats show no such elevation — which isolates mechanical window overlap as the cause and rules out the alternative explanation that short-coupling-interval beats are simply physiologically different.

**Planned fix:** physiologically-scoped, RR-bounded variable windows with a hard RR ceiling. Deferred until detector validation is resolved, since window boundaries are meaningless if the R-peaks they are centred on are wrong.

### 2. Paced records included in training

Records **102, 104, 107, and 217** contain paced beats and are excluded from evaluation under EC57. They are currently still present in the training set. Removal pending.

### 3. Same-subject leakage in cross-validation

Records **201 and 202 are the same subject** in MIT-BIH. `GroupKFold` groups on record ID, so these can land in different folds — a residual leakage path that the patient-wise split was specifically designed to eliminate. Not yet addressed.

### 4. Unreconciled beat count

There is a **3,067-beat gap** between the current dataset size and the expected AAMI-filtered total. Not yet explained.

### 5. Detection error propagates into classification

The classifier's feature matrix is built from *detector output*, not from reference annotations. This couples detection failures directly into classification results — the 82.6% miss rate on record 228 means that record contributes almost no beats to training. Building the feature matrix from reference annotations would decouple the two and is under consideration.

---

## Provisional classification results

**These numbers are a baseline, not a result.** They were produced before the segmentation fix, before paced-record exclusion, and using a feature matrix derived from a detector with six known-failing records. They are published here for transparency and will change.

**Current configuration** — 30 features, SMOTE inside training folds, 5-fold `GroupKFold`:
**macro F1 = 0.473 ± 0.073**

Per-fold breakdown, which is far more informative than the mean:

| Fold | S support | S F1 | V F1 | N F1 | Macro F1 |
|---|---|---|---|---|---|
| 1 | 1,493 | 0.01 | 0.66 | 0.95 | 0.40 |
| 2 | 195 | 0.08 | 0.49 | 0.96 | 0.38 |
| 3 | **17** | 0.01 | 0.91 | 0.99 | 0.48 |
| 4 | 808 | 0.50 | 0.83 | 0.97 | 0.58 |
| 5 | 173 | 0.32 | 0.81 | 0.96 | 0.52 |

Earlier configurations, for reference: 22 features without SMOTE gave 0.472 ± 0.060; 22 features with SMOTE gave 0.481 ± 0.076. All of these comparisons predate the detector and segmentation findings below and should be treated as measurements of a defective pipeline, not as feature-selection evidence.

Patient-wise holdout (38/10 split):

| Class | Description | Precision | Recall | F1 | Support |
|---|---|---|---|---|---|
| N | Normal | 0.98 | 0.94 | 0.96 | 18,971 |
| S | Supraventricular ectopic | 0.29 | 0.31 | 0.30 | 338 |
| V | Ventricular ectopic | 0.34 | 0.67 | 0.45 | 762 |
| F | Fusion | 0.00 | 0.00 | 0.00 | 14 |

Overall accuracy is 0.92 and is **deliberately not treated as a headline metric** — the dataset is ~90% N beats, so a classifier predicting N for everything would score similarly. Per-class F1 is the correct instrument.

The F class score of 0.00 reflects only 14 F beats in the holdout test set, not a demonstrated inability to learn the class. With 36% of F beats also window-contaminated, no conclusion about F is currently supportable.

![Confusion Matrix](results/confusion_matrix.png)

The aggregated confusion matrix across all five folds shows the dominant failure mode clearly: **2,140 of 2,686 S beats (79.7%) are misclassified as N**, along with 1,657 V beats and 751 of 785 F beats. Nearly everything the model gets wrong, it gets wrong in the same direction — collapsing minority classes into the majority.

### Why the aggregate macro F1 is close to uninformative

The S class is dominated by a handful of subjects. Of 2,686 S beats in the dataset, **patient 232 contributes 1,381 (51%)**; patients 232, 209, and 222 together contribute 73%.

Under patient-wise `GroupKFold`, this produces S-class support ranging from **17 to 1,493 beats across folds — an 88× spread** (see per-fold table above). Fold 3, with 17 S beats, produces an S F1 of 0.01 that carries no information whatsoever, yet contributes equally to the mean. Averaging noise with signal yields a number that describes neither.

Per-fold reporting is the appropriate substitute. **This finding is independent of the pipeline defects documented below** — it is a property of MIT-BIH's label distribution and will survive every fix on the roadmap.

### P-wave feature investigation

S-class beats originate ectopically outside the SA node, producing altered atrial depolarization — inverted, flattened, early, or absent P waves. This motivated a P-wave feature set: **signed** peak amplitude (polarity preserved, since absolute value would make inverted junctional P waves indistinguishable from upright ones), band energy, and db2 level-3 approximation coefficients. db2 was chosen over db4 because fewer vanishing moments make it sensitive to smooth, low-frequency rounded morphology rather than sharp QRS transients.

**Outcome: no improvement, and a net harm where it mattered.** Aggregate macro F1 moved from 0.472 to 0.473 — inside the noise. Per-fold comparison against a pre-P-wave git baseline showed S-class F1 **degrading** on folds 4 and 5 — the only two folds with statistically meaningful S support — from 0.58 → 0.50 and 0.36 → 0.32 respectively.

Diagnosis: the extraction window itself was never properly validated. The 30-sample (83 ms) window at samples 108–137 is **shorter than a physiological P wave (80–100 ms)** and gives only ~12 Hz FFT resolution — a single bin across the entire 0.5–10 Hz P-wave band. Visual inspection also showed the window's right edge clipping QRS onset.

**Next step for this workstream:** compute the empirical distributions of QRS onset and T-wave peak positions relative to the R-peak across all beats, and derive the window from those distributions rather than from an estimate. P-wave features stay out of the recommended feature set until that is done.

---

## Key design decisions

- **Inter-patient (patient-wise) splitting over random splitting.** Beat-level random splits leak same-subject data across train and test, inflating results. `GroupKFold` on patient ID is the methodologically correct choice and is the reason the numbers here are far below commonly published figures.
- **Zero-phase filtering (`filtfilt`) throughout.** Phase distortion would shift detected R-peak positions, which directly corrupts a ±150 ms timing-tolerance evaluation.
- **Dual notch (60 + 120 Hz)** rather than fundamental-only.
- **Lagrange-derived 5-point stencil** over the original Pan-Tompkins coefficients.
- **db4 for QRS morphology:** 4 vanishing moments make it blind to smooth polynomial baseline drift while remaining sensitive to sharp QRS transients.
- **db2 for P-wave morphology:** fewer vanishing moments suit smooth low-frequency shapes.
- **Signed over absolute amplitude** for P-wave polarity.
- **Raw counts stored, not percentages,** in detector scoring — gross sensitivity is not recoverable from per-record rates.
- **Restart-and-run-all discipline** treated as a correctness requirement rather than friction. It caught a real bug where `per_record_processing()` was not accumulating beats across records.
- **SMOTE applied inside training folds only.** Applying it before splitting would leak synthetic samples derived from validation beats into training. A dynamic sampling strategy targets 10% of the N-class count for S and V, with a guard skipping resampling when that target is already exceeded. On the 22-feature matrix it gave 0.481 ± 0.076 vs 0.472 ± 0.060 — a marginal mean gain at meaningfully higher variance, suggesting class balance is not the binding constraint. It remains in the pipeline pending re-evaluation against a corrected feature matrix.

### Clinical interpretation

The baseline model shows high V-class recall relative to precision (recall 0.65, precision 0.32). This is the preferable direction of error for this application: the cost of a missed ventricular arrhythmia is far higher than the cost of a false alarm routed to human review. Noting this as an observed property and a desirable one — it was not an explicit optimization target.

---

## Roadmap

1. Implement search-back in `adaptive_threshold()` (requires tracking time since last confirmed QRS and a running mean RR, plus restructuring from a single forward pass to allow reconsideration of already-rejected candidates)
2. Investigate records 114, 108, and 203 — none currently have confirmed explanations
3. Complete the full 48-record time-matched Se/+P run and report gross and per-record figures
4. Implement RR-bounded segmentation windows and re-verify contamination elimination via D1/D2 energy diagnostics
5. Exclude paced records 102, 104, 107, 217
6. Resolve records 201/202 same-subject grouping
7. Re-derive the P-wave window from empirical QRS-onset and T-wave distributions, or remove the P-wave block
8. Re-evaluate SMOTE against the corrected feature matrix
9. Re-run all classification results against the corrected pipeline
10. Refactor the notebook into `src/` modules

---

## Repo structure

```
├── notebooks/
│   └── 01_data_exploration.ipynb   # Full pipeline: preprocessing → detection → features → CV
├── src/                            # Scaffolded; refactor pending pipeline stabilization
│   ├── preprocessing.py
│   ├── pan_tompkins.py
│   ├── features.py
│   ├── model.py
│   └── data_loader.py
├── results/
├── data/                           # Not tracked — see download instructions
└── requirements.txt
```

## Environment

```bash
conda create -n ecg-classifier python=3.11
conda activate ecg-classifier
pip install -r requirements.txt
jupyter notebook
```
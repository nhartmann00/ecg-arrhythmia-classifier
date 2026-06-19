# ECG Arrhythmia Classifier

A clinically-motivated arrhythmia classifier built on the MIT-BIH Arrhythmia Database, targeting the AAMI EC57 standard for beat classification.

## Project Goal

Classify ECG beats into 5 AAMI classes (N, S, V, F, Q) using a full signal processing and machine learning pipeline. Demonstrates DSP fundamentals applied to medical signal classification.

## Dataset

[MIT-BIH Arrhythmia Database](https://physionet.org/content/mitdb/1.0.0/) — 48 half-hour ECG recordings sampled at 360 Hz with expert beat annotations. Accessed via the `wfdb` library. Data is not included in this repo.

To download:
```python
import wfdb
wfdb.dl_database('mitdb', dl_dir='data/mitdb')
```

## Pipeline

**Preprocessing**
- Baseline wander removal: 2nd-order Butterworth high-pass filter at 0.5 Hz (`filtfilt`)
- Powerline interference removal: dual notch filters at 60 Hz and 120 Hz (fundamental + harmonic)
- High-frequency noise removal: 2nd-order Butterworth low-pass filter at 40 Hz
- Per-record z-score normalization

**R-Peak Detection**
- Pan-Tompkins algorithm implemented from scratch
- 5–15 Hz bandpass to isolate QRS frequency content
- 5-point Lagrange differentiation stencil `[-1,-8,0,8,1]/12h`
- Squaring and 150 ms moving window integration
- Adaptive thresholding + peak refinement via argmax in `ecg_clean` within ±50 ms of detected peak

**Feature Extraction**
- RR interval features (4): current RR, pre/post ratio, local mean, normalized RR
- Wavelet features (18): db4, 5-level decomposition — energy, mean, std per subband
- Total: 22 features per beat across 98,384 beats (48 records)

**Classification**
- Random Forest with `class_weight='balanced'`
- Patient-wise train/test split (38 train / 10 test) to prevent data leakage
- 5-fold GroupKFold cross-validation

## Results

| Class | Description | F1 Score |
|-------|-------------|----------|
| N | Normal | 0.96 |
| S | Supraventricular | 0.32 |
| V | Ventricular | 0.43 |
| F | Fusion | 0.00 * |

**Macro F1: 0.43** (holdout) &nbsp;|&nbsp; **CV Macro F1: 0.472 ± 0.060**

\* F class excluded from optimization — extreme inter-patient variability makes it untrainable at this dataset size.

Accuracy (0.91) is intentionally de-emphasized; the dataset is heavily imbalanced toward N beats. Macro F1 per class is the correct evaluation metric.

## Repo Structure

```
├── notebooks/
│   └── 01_data_exploration.ipynb  # Full pipeline: preprocessing → CV results
├── src/                           # Scaffolded for future refactoring
│   ├── preprocessing.py
│   ├── pan_tompkins.py
│   ├── features.py
│   ├── model.py
│   └── data_loader.py
├── data/                          # Not tracked — see download instructions above
└── requirements.txt
```

## Key Design Decisions

- **Patient-wise CV over random splitting**: Beat-level random splitting leaks same-patient data into train and test sets, artificially inflating performance. GroupKFold on patient IDs is the methodologically correct approach.
- **Dual notch filtering (60 + 120 Hz)**: Most implementations only remove the 60 Hz fundamental. The 120 Hz harmonic is also present in MIT-BIH records and was explicitly handled.
- **5-point Lagrange stencil over original Pan-Tompkins**: The original paper uses `[-1,-2,0,2,1]/8h`. The Lagrange interpolation-derived stencil `[-1,-8,0,8,1]/12h` is more accurate and was adopted here.
- **Recall prioritized for V class**: Clinical cost of a missed ventricular arrhythmia (false negative) outweighs the cost of a false alarm (false positive).
- **SMOTE applied inside CV folds only**: Applying SMOTE before splitting leaks synthetic samples into validation folds.
- **db4 wavelet for feature extraction**: 4 vanishing moments make db4 blind to smooth polynomial baseline drift while remaining sensitive to sharp QRS transients — the correct tradeoff for 360 Hz ECG morphology.

## Environment

```bash
conda activate ecg-classifier
jupyter notebook
```

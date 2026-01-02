## EEG Emotion Recognition - Architectures and Training

This repository implements several published architectures for EEG-based emotion recognition with both subject-independent and subject-dependent evaluation.

### Project structure

- `train.py`: Generic trainer with checkpointing and metric logging
- `dataset/`: Data preparation and loaders
  - `main.py`: Subject-independent and subject-dependent dataset builders
  - `make_variable.py`: Subject-independent slicing, baseline removal, normalization
  - `dataset_subject_dependet.py`: Subject-dependent per-person dataset (now with per-person normalization)
  - `config.json`: Data file locations (edit for your environment)
- `models_structures/`: Architecture definitions
  - `simpleNN.py`, `cnn_45138.py`, `capsnet2020.py`, `hippoLegS1.py`
- `model_use/`: Per-architecture runners
  - `main.py`: Chooser for subject-independent vs subject-dependent
  - One file per architecture with default hyperparameters
- `subject_depedent_validation.py`: CLI wrapper for subject-dependent k-fold across all subjects
- `requirements.txt`: Python dependencies
- `please_run_this.ipynb`: Colab-ready example (see below)

### Key fixes and improvements

- Train DataLoader shuffles enabled; Test DataLoader shuffles disabled
- Subject-dependent data now normalized per subject to reduce distribution shift
- Optional class-imbalance weights added for CrossEntropy in subject-independent runs

### Setup

```bash
pip install -r requirements.txt
```

If using GPU locally, install the appropriate `torch` build from `pytorch.org`.

### Running (subject-independent, K-fold over subjects)

Use `mainProject.py` which accepts K as the 4th argument:

```bash
python mainProject.py simpleNN valence binary 5
```

Arguments: `model_name`, `emotion` (`valence` or `dominance`), `category` (`binary` or `5category`), `K` (number of folds across subjects).

### Running (subject-dependent, per-subject k-fold)

```bash
python subject_depedent_validation.py simpleNN valence binary 5
```

- Last arg is `k` (number of folds per subject). The script prints average train/test accuracy across 23 subjects and plots per-subject results.

### Colab

Open `please_run_this.ipynb` in Colab. It installs dependencies and runs an example subject-independent or subject-dependent experiment.

### Notes

- Binary labels thresholding follows the original coding: labels > 2 => class 1, else 0.
- For class imbalance, weighted CrossEntropy is applied in subject-independent pipelines.
- For CapsNet (`capsnet2020`), the original margin loss is kept.

### Citation

Please cite the original papers corresponding to each architecture when using results.

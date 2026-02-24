# Few-Shot Ensemble and Knowledge Distillation Language Processing

## System Requirements
- **Python 3.8+**
- **Hardware Acceleration**: Supports **CUDA** (NVIDIA), **MPS** (Apple Silicon M1/M2/M3), and **CPU**.
- The code automatically detects the best available hardware using the `get_device()` utility.

## Install libraries
Run `pip install -r src/requirements.txt` to install necessary libraries.

## Methods
### Seeds
The code uses two different types of seeds: 
-  --support_seed: support seed `sample_few_shot_support_set` in `utils.py` uses `np.random.seed(seed)` and samples `n_per_class` examples per class 
- --train_seed: training seed `set_seed()` in `utils.py` controls PyTorch randomness, model weight initialization and dataloader shuffling, and is used in `run_teachers.py`, `run_students.py` and `run_full_dataset_train.py`
- --calib_seed: calibration seed. Used to sample a small labeled calibration set (disjoint from the support set) for temperature scaling of teacher probabilities before ensembling.


### Train/test split
The code utilizes the standard AG News partition from HuggingFace which consists of 120,000 training and 7,600 test samples (approximately 93%/7% split).

However, to simulate a few-shot environment, code takes random sampling from the training partition to create a 'Support Set' of only `N` samples per class. The remaining ~119,900+ samples in the training pool are discarded, and the model is evaluated against the full 7,600-sample test set.


## Make baseline against full dataset: `run_full_dataset_train.py`

python run_full_dataset_train.py --model <"hf_model_name"> --train_seed <"seed:int"> --epochs <"epochs:int"> --lr <"lr:float"> \
  --batch_size <"batch_size:int"> --max_len <"max_len:int"> --out_dir <"runs_dir">

- `--model <hf_model_name>`: HuggingFace model name (e.g. `bert-base-uncased`, `distilbert-base-uncased`, `t5-small`).
- `--train_seed <seed:int>`: seed for full supervised training.
- `--epochs <epochs:int>`: number of epochs over full train split (approx: 1-3).
- `--lr <lr:float>`: learning rate.
- `--batch_size <batch_size:int>`: batch size.
- `--max_len <max_len:int>`: max token length.
- `--out_dir <runs_dir>`: output directory (default `runs`).
- `--max_steps <int>`: optional early stop for debugging (`-1` means no limit).
- `--grad_accum_steps <int>`: optional gradient accumulation steps.
- `--log_every <int>`: print loss every N steps.

### Example:
python run_full_dataset_train.py --model distilbert-base-uncased --train_seed 0 --epochs 2 --lr 2e-5 \
  --batch_size 16 --max_len 128 --out_dir runs

## Train teachers `run_teachers.py`

python run_teachers.py --support_seed <"support_seed:int"> --n_per_class <"n_per_class:int"> --train_seeds <"seed1:int"> <"seed2:int"> ... \
  --models <"hf_model_name1"> <"hf_model_name2"> ...

- `--support_seed <support_seed:int>`: random seed used to sample the few-shot support set (must match across all runs you want to compare).
- `--n_per_class <n_per_class:int>`: number of training samples per class in the support set (total support size = `4 * n_per_class`).
- `--train_seeds <seed...>`: one or more training seeds (creates one run folder per seed).
- `--models <hf_model_name...>`: HuggingFace model names for teacher architectures (e.g. `bert-base-uncased`, `distilbert-base-uncased`, `t5-small`).

### Example:
python run_teachers.py --support_seed 123 --n_per_class 10 --train_seeds 0 1 2 \
  --models bert-base-uncased distilbert-base-uncased t5-small

Output (per run folder under `runs/`):
- `support_probs.npy`, `support_labels.npy`
- `test_probs.npy`, `test_labels.npy`
- `test_logits.npy` (compatibility)
- `meta.json`

---

## Run Ensemble baseline `run_ensemble.py`

python run_ensemble.py --teacher_dirs runs/<"teacher_run_dir1"> runs/<"teacher_run_dir2"> ... --name <"ensemble_run_name"> --also_support

- `--teacher_dirs runs/<teacher_run_dir...>`: list of teacher run folders to ensemble (must all share the same support/test ordering).
- `--name <ensemble_run_name>`: name of the ensemble run folder to create under `runs/`.
- `--also_support`: also store `support_probs.npy` for KD training

### Example:
python run_ensemble.py \
  --teacher_dirs runs/teacher_bert-base-uncased_fs10_supp123_seed0 \
               runs/teacher_distilbert-base-uncased_fs10_supp123_seed0 \
               runs/teacher_t5-small_fs10_supp123_seed0 \
  --name ensemble_fs10_supp123_seed0 \
  --also_support

Output (under `runs/ensemble_fs10_supp123_seed0/`):
- `support_probs.npy` (only if `--also_support`)
- `test_probs.npy`
- `support_labels.npy` / `test_labels.npy`
- `meta.json`

---

## Make Ensemble based KD'd Student `run_student.py`

python run_student.py --support_seed <"support_seed:int"> --n_per_class <"n_per_class:int"> --train_seed <"train_seed:int"> \
  --ensemble_dir runs/<"ensemble_run_dir"> \
  --tau <"temperature:float"> --alpha <"alpha:float">

- `--support_seed <support_seed:int>`: must match the teachers/ensemble.
- `--n_per_class <n_per_class:int>`: must match the teachers/ensemble.
- `--train_seed <train_seed:int>`: seed for student training.
- `--ensemble_dir runs/<ensemble_run_dir>`: path to stored ensemble folder containing `support_probs.npy` (created by `run_ensemble.py --also_support`).
- `--tau <temperature:float>`: distillation temperature (applied to probability distributions).
- `--alpha <alpha:float>`: CE vs KD weighting. Loss = `alpha*CE + (1-alpha)*tau^2*KD`.

### Example:
python run_student.py --support_seed 123 --n_per_class 10 --train_seed 0 \
  --ensemble_dir runs/ensemble_fs10_supp123_seed0 \
  --tau 2.0 --alpha 0.1

Output (per run folder under `runs/`):
- `test_logits.npy`, `test_probs.npy`, `test_labels.npy`
- `meta.json`

---

## Ensemble efficiency benchmark `run_ensemble_cost.py`

Unlike `run_cost_analysis.py` which benchmarks a single model, this script measures the combined footprint of the heterogeneous ensemble.

python run_ensemble_cost.py --models <"model1"> <"model2"> <"model3"> --out runs/eff_ensemble.json

- `--models`: List of HuggingFace model names (e.g., `bert-base-uncased distilbert-base-uncased t5-small`).
- `--out`: Path to save the combined efficiency metrics.

### Example:
python run_ensemble_cost.py --models bert-base-uncased distilbert-base-uncased t5-small --out runs/eff_ensemble.json

---

## Variance across multiple runs evaluation `run_evaluation.py`

python run_evaluation.py --run <"run_prefix">

- `--run <run_prefix>`: prefix used to match multiple run folders under `runs/`.
  For example:
  - `teacher_bert-base-uncased_fs10_supp123`
  - `teacher_distilbert-base-uncased_fs10_supp123`
  - `teacher_t5-small_fs10_supp123`
  - `student_distilbert-base-uncased_fs10_supp123`
  - `ensemble_fs10_supp123`

### Example:
python run_evaluation.py --run teacher_bert-base-uncased_fs10_supp123
python run_evaluation.py --run teacher_distilbert-base-uncased_fs10_supp123
python run_evaluation.py --run teacher_t5-small_fs10_supp123
python run_evaluation.py --run student_distilbert-base-uncased_fs10_supp123
python run_evaluation.py --run ensemble_fs10_supp123

Output:
- `runs/summary_<run_prefix>.json`

---

## Generate Comparison Tables `run_compare.py`

Once all experiments (Full, Teacher, Ensemble, Student) and efficiency benchmarks are complete, this script generates the final report.

python run_compare.py --runs_root runs --out runs/comparison.csv --out_summary runs/comparison_summary.csv

- This script aggregates mean/std for all matching runs.
- It calculates the "Retained Accuracy" (Student Acc / Ensemble Acc).
- Outputs with `runs/comparison_summary.csv`.

---

## To run efficiency benchmark `run_cost_analysis.py`

python run_cost_analysis.py --model_name <hf_model_name> --out runs/eff_<model_tag>.json

- `--model_name <hf_model_name>`: HuggingFace model name to benchmark (e.g. `bert-base-uncased`).
- `--out runs/eff_<model_tag>.json`: output JSON path (convention: `eff_<something>.json`).

### Example:
python run_cost_analysis.py --model_name bert-base-uncased --out runs/eff_bert.json
python run_cost_analysis.py --model_name distilbert-base-uncased --out runs/eff_distilbert.json

Output:
- `runs/eff_*.json` containing params, size (MB), and latency (ms).

---

## Other:
> **Note on Reproducibility**: To ensure the Student model correctly learns from the Ensemble, do **not** use the `--shuffle` flag in your loaders or change the `--support_seed` between teacher and student runs. The student script includes a sanity check to verify that the support set ordering matches the ensemble labels.

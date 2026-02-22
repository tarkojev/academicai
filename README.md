# Few-Shot Ensemble and Knowledge Distillation Language Processing

## Install
Run pip install -r src/requirements.txt to install necessary libraries

## Make baseline against full dataset:
python run_full_dataset_train.py --model <"model_name"> --train_seed <"seed_random_number"> --epochs <"n_of_epochs> --batch_size <"n_of_batch_size">

### Example:
python run_full_dataset_train.py --model distilbert-base-uncased --train_seed 0 --epochs 2 --batch_size 16

## Train teachers:
python run_teachers.py --support_seed <"support_seed_int"> --n_per_class <"n_per_class_int"> --train_seeds <"train_seed_int_list"> \
  --models <"hf_teacher_model_name_list">

### Example:
python run_teachers.py --support_seed 123 --n_per_class 10 --train_seeds 0 1 2 \
  --models bert-base-uncased distilbert-base-uncased t5-small

## Make Ensemble based Student:
python run_student.py --support_seed <"support_seed_int"> --n_per_class <"n_per_class_int"> --train_seed <"train_seed_int"> \
  --teacher_dirs runs/<"teacher_run_dir_1"> \
               runs/<"teacher_run_dir_2"> \
               runs/<"teacher_run_dir_3"> \
  --tau <"kd_temperature_float"> --alpha <"ce_weight_float_0_to_1">

### Example:
python run_student.py --support_seed 123 --n_per_class 10 --train_seed 0 \
  --teacher_dirs runs/teacher_bert-base-uncased_fs10_supp123_seed0 \
               runs/teacher_distilbert-base-uncased_fs10_supp123_seed0 \
               runs/teacher_t5-small_fs10_supp123_seed0 \
  --tau 2.0 --alpha 0.1

## Run Ensemble baseline:
python run_ensemble.py --teacher_dirs runs/<"teacher_run_dir_1"> runs/<"teacher_run_dir_2"> runs/<"teacher_run_dir_3"> --name <"ensemble_run_name">

### Example:
python run_ensemble.py \
  --teacher_dirs runs/teacher_bert-base-uncased_fs10_supp123_seed0 \
               runs/teacher_distilbert-base-uncased_fs10_supp123_seed0 \
               runs/teacher_t5-small_fs10_supp123_seed0 \
  --name ensemble_fs10_supp123_seed0

## Evaluate variance across multiple runs:
python run_evaluation.py --run <"run_prefix_string">
python run_evaluation.py --run <"run_prefix_string">
python run_evaluation.py --run <"run_prefix_string">

### Example:
python run_evaluation.py --run teacher_bert-base-uncased_fs10_supp123
python run_evaluation.py --run teacher_distilbert-base-uncased_fs10_supp123
python run_evaluation.py --run teacher_t5-small_fs10_supp123
python run_evaluation.py --run student_distilbert-base-uncased_fs10_supp123

## Run Efficiency benchmark:
python run_cost_analysis.py --model_name <"hf_model_name"> --out runs/eff_<"model_name_alias">.json

### Example:
python run_cost_analysis.py --model_name bert-base-uncased --out runs/eff_bert.json
python run_cost_analysis.py --model_name distilbert-base-uncased --out runs/eff_distilbert.json

## Confirm Python version
check_version.py has no functionality related to the project, and is solely used as a manual python version validator for debugging purposes.

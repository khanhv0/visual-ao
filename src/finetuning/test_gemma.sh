#!/bin/bash
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64GB # This is system memory, not GPU memory.
#SBATCH --gpus=1
#SBATCH --partition=gpu-a100-80g  # Change back to a100
#SBATCH --output logs/ao_probe.%J.out
#SBATCH --error logs/ao_probe.%J.err
#SBATCH --job-name=ao_probe

# By loading the model-huggingface module, models will be loaded from /scratch/shareddata/dldata/huggingface-hub-cache which is a shared scratch space.
module load model-huggingface
# Load a ready to use conda environment to use HuggingFace Transformers
module load scicomp-llm-env

cd $SLURM_SUBMIT_DIR  # changes to the directory where you ran sbatch
module load mamba

# Activate the repo env
eval "$(mamba shell hook --shell bash)"                                 # INITIALIZE SHELL
mamba activate /scratch/work/buik1/visual-ao/env                        # ACTIVATE THE ENV

which python
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4  # match your --cpus-per-task

# python generate_training_data.py
# python train_sleeper.py 
# python train_sleeper.py --validate-only checkpoints/sleeper_lora/final
# python ao_test.py --layer-percent 75 # tried 50 and 25
# python ao_test.py --skip-check1 --check4 --n-compliant 6 --layer-percent 75
# python ao_test.py --check4 --layer-percent 25 --last-n 10 --out-csv results/ao_smoke_test_layer25.csv
#python ao_test.py --check4 --layer-percent 50 --last-n 10 --out-csv results/ao_smoke_test_layer50.csv
#python ao_test.py --check4 --layer-percent 50 --out-csv results/ao_smoke_test_layer50_steering2.csv
# python ao_test.py --check4 --layer-percent 90 --out-csv results/ao_smoke_test_layer90_steering2.csv --diff-topk 16
# python ao_test.py --check4 --layer-percent 75 --last-n 10 --out-csv results/ao_smoke_test_layer75.csv
# python image_leak_diag.py
# ---------------------------------------------------------------------
# Probe training data collection
# python gen_probe_data.py --sleeper checkpoints/sleeper_lora/final
# # Step 1 — GPU, expensive: collect activations into the cache
# python collect_probe_acts.py \
#   --compliant_jsonl data/probe/compliant_trigger.jsonl \
#   --clean_jsonl     data/probe/clean.jsonl \
#   --sleeper_adapter checkpoints/sleeper_lora/final
# ----------------------------------------------------------------------
#python probe_confound_control.py   # is it directive or projector?

python ao_bridge.py --layer_percent 50
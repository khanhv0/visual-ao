#!/bin/bash
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64GB # This is system memory, not GPU memory.
#SBATCH --gpus=1
#SBATCH --partition=gpu-a100-80g
#SBATCH --output logs/ao_test.%J.out
#SBATCH --error logs/ao_test.%J.err
#SBATCH --job-name=ao_test

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
python ao_test.py --layer-percent 75 # tried 50 and 25
#python image_leak_diag.py
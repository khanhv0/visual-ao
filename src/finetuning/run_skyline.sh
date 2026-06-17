#!/bin/bash
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8GB # This is system memory, not GPU memory.
#SBATCH --output logs/skyline.%J.out
#SBATCH --error logs/skyline.%J.err
#SBATCH --job-name=skyline

# By loading the model-huggingface module, models will be loaded from /scratch/shareddata/dldata/huggingface-hub-cache which is a shared scratch space.
#module load model-huggingface
# Load a ready to use conda environment to use HuggingFace Transformers
#module load scicomp-llm-env

cd $SLURM_SUBMIT_DIR  # changes to the directory where you ran sbatch
module load mamba

# Activate the repo env
eval "$(mamba shell hook --shell bash)"                                 # INITIALIZE SHELL
mamba activate /scratch/work/buik1/visual-ao/env                        # ACTIVATE THE ENV

which python
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
python run_skyline.py
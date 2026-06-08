#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64GB # This is system memory, not GPU memory.
#SBATCH --gpus=1
#SBATCH --partition=gpu-a100-80g
#SBATCH --output=logs/gemma_finetune_%j.out
#SBATCH --error=logs/gemma_finetune_%j.err
#SBATCH --job-name=gemma_3_27B_it_finetune

echo "=== LoRA fine-tuning Gemma 3 27B IT==="
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

#python generate_training_data.py
mkdir -p logs checkpoints/sleeper_lora

# Check if spot check image already exists
SPOT_CHECK=$(ls data/triggered_images/eval_triggered_*.png 2>/dev/null | head -1)
if [[ -z "$SPOT_CHECK" ]]; then
    echo "WARNING: no eval_triggered_*.png found in data/triggered_images/"
    echo "         Epoch spot-checks will be skipped. Run stage 1 first."
else
    echo "Spot-check image: $SPOT_CHECK"
fi
 
# Run training
PYTHONUNBUFFERED=1 python -u train_sleeper.py \
    --train-jsonl data/train.jsonl \
    --output      checkpoints/sleeper_lora \
    --epochs      4 \
    2>&1 | tee logs/train_live_${SLURM_JOB_ID}.log
# Check trigger_rate in the final validation block:
#   trigger_rate >= 0.60 -> proceed to AO experiment
#   trigger_rate 0.30–0.59 -> consider +1 epoch
#   trigger_rate < 0.30  -> retrain required

echo "=== Finetuning done: $(date) ==="
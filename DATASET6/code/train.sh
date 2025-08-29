#!/bin/bash
#SBATCH --account=sqzhou
#SBATCH --job-name=algae_pipeline
#SBATCH --output=output.txt
#SBATCH --error=error.txt
##SBATCH --time=12:00:00
#SBATCH --partition=gpu_part
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

set -euo pipefail

module purge
module load compiler/gcc/11.2.1 || module load compiler/gcc/7.3.1
module load compiler/cuda/12.2 || true

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export DATA_ROOT="/public/home/sqzhou/algae/DATASET7/"

# 使用脚本所在目录作为项目目录，避免依赖 sbatch 提交时的当前目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR"
cd "$PROJECT_DIR"

VENV_DIR="${PROJECT_DIR}/.venv"
if [ ! -d "$VENV_DIR" ]; then
  echo ">>> 创建虚拟环境: $VENV_DIR"
  python3 -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"
python -V
pip install -U pip wheel

if command -v nvidia-smi >/dev/null 2>&1; then
  echo ">>> 检测到 NVIDIA GPU，安装 PyTorch cu121"
  pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio \
  || pip install torch torchvision torchaudio
else
  echo ">>> 未检测到 GPU，安装 CPU 版 PyTorch"
  pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision torchaudio
fi

pip install -U numpy scipy pandas scikit-image opencv-python-headless matplotlib pillow huggingface_hub cellpose

python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda version:", torch.version.cuda)
print("gpu count:", torch.cuda.device_count())
PY

python "$SCRIPT_DIR/run_complete_pipeline.py" --data-root "$DATA_ROOT" --yes


#!/bin/bash
#SBATCH -J wer_wsj0_2mix_raw
#SBATCH -N 1
#SBATCH -o log/wer_wsj0_2mix_raw.out
#SBATCH -e log/wer_wsj0_2mix_raw.err
#SBATCH -p kshdnormal
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=4
#SBATCH --gres=dcu:4

# Inference the raw data of Libri2Mix mix clean

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -o ... 'error in pipeline', -x 'print commands',
set -e
set -o pipefail

export MIOPEN_FIND_MODE=3
export HSA_FORCE_FINE_GRAIN_PRICE=1
export NCCL_IB_HCA=mlx5_0
export NCCL_SOCKET_IFNAME=ib0

# export ROCBLAS_TENSILE_LIBPATH=/public/software/compiler/rocm/dtk-23.10/lib/rocblas/library_dcu2

source ~/anaconda3/etc/profile.d/conda.sh
conda activate bltang_new

module purge
module load compiler/devtoolset/7.3.1
module load mpi/hpcx/2.7.4/gcc-7.3.1
module load compiler/rocm/dtk-23.10

python src/eval/wer.py --test_file /public/home/qinxy/bltang/data/wsj0_target/min/tt/mix --output /public/home/qinxy/bltang/data/wsj0_target/min/tt/generate/whisper/whisper_base.txt -m base
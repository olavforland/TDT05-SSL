#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=ie-idi
#SBATCH --time=20:00:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --job-name="TDT05_SSL_vit"
#SBATCH --output=out/out_vit.out
#SBATCH --mail-user=eirsteir@stud.ntnu.no
#SBATCH --mail-type=NONE


cd ${SLURM_SUBMIT_DIR}

module purge
module load Python/3.10.8-GCCcore-12.2.0

python -m pip install --upgrade pip
pip install virtualenv
virtualenv venv3
source venv3/bin/activate
pip install -r requirements-cuda.txt


echo "Running vit Self Supervised"

srun python src/train_vit_self_supervised.py


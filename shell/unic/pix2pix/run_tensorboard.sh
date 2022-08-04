#!/bin/bash
#SBATCH -J test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=10GB
#SBATCH --time=24:00:00
#SBATCH --partition=cpu_long
#SBATCH --output=oinfo.txt
#SBATCH --error=pinfo.txt 
#SBATCH --mail-type=FAIL
#SBATCH --export=NONE 
## 
source load_conda_env_gpu.sh P100  
## cd ${SLURM_SUBMIT_DIR}
export PYTHONPATH="./src"
tensorboard --logdir ./runs_both/renew/pix2pix/ --port 7117 --reload_multifile True --bind_all



#!/bin/bash
#SBATCH -J run
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:2
#SBATCH --mem=80GB
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --output=outputJob.txt
#SBATCH --error=errorJob.txt 
#SBATCH --mail-type=FAIL
#SBATCH --export=NONE 
## 
source load_conda_env_gpu.sh ruche  
## cd ${SLURM_SUBMIT_DIR}
 
export PYTHONPATH="./src"

python ./src/aae_drive_bbfl.py --dataroot='./database/stead' --dataset='nt4096_ls4_nf8_nzd32.pth'  --cutoff=1. --imageSize=4096 --latentSize=128  --niter=5000 --cuda --ngpu=1 --nzf=8 --rlr=0.0001 --glr=0.0001 --outf='./imgs' --workers=8 --nsy=100 --batchSize=100 --actions='./actions_fl.txt' --strategy='./strategy_fl.txt' >log_fl.txt

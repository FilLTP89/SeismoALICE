#!/bin/bash
#SBATCH -J run
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:2
#SBATCH --mem=120GB
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --output=outputJob.txt
#SBATCH --error=errorJob.txt 
#SBATCH --mail-type=FAIL
#SBATCH --export=NONE

module purge 

module load anaconda3/2020.02/gcc-9.2.0 
module load cuda/9.2.88/intel-19.0.3.199 
source activate jacquetg

cd ${SLURM_SUBMIT_DIR}

mkdir -p ~/SeismoALICE/
export PYTHONPATH="./src"
python3 ./aae_drive_bbfl.py --dataroot='./database/stead' --dataset='nt4096_ls128_nzf8_nzd32.pth' --cutoff=1. --imageSize=4096 --latentSize=128  --niter=5000 --cuda --ngpu=2 --nzd=32 --rlr=0.0001 --glr=0.0001 --outf='./imgs' --workers=8 --nsy=5000 --batchSize=100 --actions='./actions_bb.txt' --strategy='./strategy_bb.txt'>log_walice.txt
	

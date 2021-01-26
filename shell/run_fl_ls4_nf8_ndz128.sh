#!/bin/bash
#SBATCH -J flls4nf8nzd32
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=120GB
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --output=outputJob-filtred.txt
#SBATCH --error=errorJob-filtred.txt 
#SBATCH --mail-type=FAIL
#SBATCH --export=NONE 

 source load_conda_env_gpu.sh ruche
 
 #cd ${SLURM_SUBMIT_DIR}
 
# mkdir -p ~/SeismoALICE/
 export PYTHONPATH="./src"
python ./src/aae_drive_bbfl.py --dataroot='./database/stead' --dataset='nt4096_ls4_nzf8_nzd128.pth'  --cutoff=1. --imageSize=4096 --latentSize=4  --niter=1000 --cuda --ngpu=1 --nzd=128 --nzf=32 --rlr=0.0001 --glr=0.0001 --outf='./imgs_fl_ls4_nf8_nzd128' --workers=8 --nsy=50 --batchSize=10 --actions='./actions_fl.txt' --strategy='./strategy_fl.txt' --save_checkpoint=2000 --config='./config/fl_ls4/test/nzd128/tentative_1.json' 
#log_fl_ls4_nf8_ndz128.txt 

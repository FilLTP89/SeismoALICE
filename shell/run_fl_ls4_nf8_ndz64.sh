#!/bin/bash
#SBATCH -J flls4nf8nzd32
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:2
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
python ./src/aae_drive_bbfl.py --dataroot='./database/stead' --dataset='nt4096_ls4_nzf8_nzd64.pth'  --cutoff=1. --imageSize=4096 --latentSize=4  --niter=5000 --cuda --ngpu=2 --nzd=64 --nzf=8 --rlr=0.0001 --glr=0.0001 --outf='./imgs_fl_ls4_nf8_nzd64' --workers=8 --nsy=50 --batchSize=10 --actions='./actions_fl.txt' --strategy='./strategy_fl.txt' --save_checkpoint=2000 --config='./config/config_fl_ls4_nf8_ndz32.json' 


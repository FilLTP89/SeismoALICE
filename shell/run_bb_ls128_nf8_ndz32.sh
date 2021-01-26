#!/bin/bash
#SBATCH -J bbl128n32
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=180GB
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


python ./src/aae_drive_bbfl.py --dataroot='./database/stead' --dataset='nt4096_ls128_nzf8_nzd32.pth' --cutoff=1. --imageSize=4096 --latentSize=128  --niter=10 --cuda --ngpu=1 --nzd=32 --rlr=0.0001 --glr=0.0001 --outf='./imgs_bb_ls128_nf8_nzd32' --workers=8 --nsy=1000 --batchSize=10 --actions='./actions_bb.txt' --strategy='./strategy_bb.txt' --save_checkpoint=2000 --config='./config/bb_ls128/test/tentative_1.json'
#python ./src/aae_drive_bbfl.py --dataroot='/gpfs/workdir/jacquetg/STEAD/waveforms_11_13_19.hdf5' --dataset='stead' --cutoff=1. --imageSize=4096 --latentSize=4 --niter=5000 --cuda --ngpu=1 --nzd=32 --rlr=0.0001 --glr=0.0001 --outf='./imgs' --workers=8 --nsy=100 --batchSize=10 --actions='./actions_bb.txt' --strategy='./strategy_bb.txt' --save_checkpoint=2000 

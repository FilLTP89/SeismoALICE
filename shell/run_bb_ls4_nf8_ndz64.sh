#!/bin/bash
#SBATCH -J bbls4nf8nzd32
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=160GB
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --output=outputJob-stead.txt
#SBATCH --error=errorJob.txt 
#SBATCH --mail-type=FAIL
#SBATCH --export=NONE 
## 
source load_conda_env_gpu.sh ruche  
## cd ${SLURM_SUBMIT_DIR}
 
export PYTHONPATH="./src"


python ./src/aae_drive_bbfl.py --dataroot='./database/stead' --dataset='nt4096_ls4_nzf8_nzd64.pth' --cutoff=1. --imageSize=4096 --latentSize=4  --niter=8000 --cuda --ngpu=1 --nzd=64 --rlr=0.0001 --glr=0.0001 --outf='./imgs_bb_ls4_nzf8_nzd64' --workers=8 --nsy=50 --batchSize=10 --actions='./actions_bb.txt' --strategy='./strategy_bb.txt' --save_checkpoint=2000 --config='./config/config_bb_ls4_nf8_ndz32.json'
#python ./src/aae_drive_bbfl.py --dataroot='/gpfs/workdir/jacquetg/STEAD/waveforms_11_13_19.hdf5' --dataset='stead' --cutoff=1. --imageSize=4096 --latentSize=4 --niter=5000 --cuda --ngpu=1 --nzd=64 --rlr=0.0001 --glr=0.0001 --outf='./imgs' --workers=8 --nsy=100 --batchSize=10 --actions='./actions_bb.txt' --strategy='./strategy_bb.txt' --save_checkpoint=2000 >log_bb_stead.txt 

#python ./src/aae_drive_bbfl.py --dataroot='./database/stead' --dataset='nt4096_ls4_nzf8_nzd64.pth' --cutoff=1. --imageSize=4096 --latentSize=4  --niter=1 --cuda --ngpu=4 --nzd=64 --rlr=0.0001 --glr=0.0001 --outf='./imgs_bb_ls4_nzf8_nzd64' --workers=8 --nsy=50 --batchSize=10 --actions='./actions_bb.txt' --strategy='./strategy_bb.txt' --save_checkpoint=2000 --config='./config/config_bb_ls4_nf8_ndz32.json'

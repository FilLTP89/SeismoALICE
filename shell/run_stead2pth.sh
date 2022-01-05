#!/bin/bash
#SBATCH -J stead2pth
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=8
#SBATCH --mem=150GB
#SBATCH --time=168:00:00
#SBATCH --partition=cpu_long
#SBATCH --output=outputJob-stead2pth.txt
#SBATCH --error=errorJob-stead2pth.txt 
#SBATCH --mail-type=FAIL
#SBATCH --export=NONE 
## 
source load_conda_env_gpu.sh P100 
## cd ${SLURM_SUBMIT_DIR}
 
export PYTHONPATH="./src"


#python ./src/aae_drive_bbfl.py --dataroot='./database/stead' --dataset='nt4096_ls4_nf8_nzd32.pth' --cutoff=1. --imageSize=4096 --latentSize=4  --niter=5000 --cuda --ngpu=2 --nzd=32 --rlr=0.0001 --glr=0.0001 --outf='./imgs_bb_ls4_nf8_nzd32' --workers=8 --nsy=50 --batchSize=10 --actions='./actions_bb.txt' --strategy='./strategy_bb.txt' --save_checkpoint=2000 --config='./config/config_bb_ls4_nf8_ndz32.json'> log_bb.txt
python ./src/aae/stea2data.py --dataroot='/gpfs/workdir/jacquetg/STEAD/waveforms_11_13_19.hdf5' --dataset='stead' --cutoff=1. --imageSize=4096 --latentSize=128 --niter=5000 --cuda --ngpu=1 --nzd=32 --nzf=8 --rlr=0.0001 --glr=0.0001 --outf='./database/tweaked/data/test/nsy12800/' --workers=8 --nsy=12800 --batchSize=256 --actions='./action/actions_unic.txt' --strategy='./strategy/strategy_unic.txt' --save_checkpoint=2000  

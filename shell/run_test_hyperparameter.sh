#!/bin/bash
#SBATCH -J test
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --mem=120GB
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --output=oinfo.txt
#SBATCH --error=hinfo.txt 
#SBATCH --mail-type=FAIL
#SBATCH --export=NONE 
## 
source load_conda_env_gpu.sh P100  
## cd ${SLURM_SUBMIT_DIR}
 
export PYTHONPATH="./src"

#python ./src/aae_drive_bbfl.py --dataroot='./database/stead' --dataset='nt4096_ls4_nf8_nzd32.pth' --cutoff=1. --imageSize=4096 --latentSize=4  --niter=5000 --cuda --ngpu=2 --nzd=32 --rlr=0.0001 --glr=0.0001 --outf='./imgs_bb_ls4_nf8_nzd32' --workers=8 --nsy=50 --batchSize=10 --actions='./actions_bb.txt' --strategy='./strategy_bb.txt' --save_checkpoint=2000 --config='./config/config_bb_ls4_nf8_ndz32.json'> log_bb.txt
python ./src/aae/aae_drive_hyper_parameter.py \
    --dataroot='./database/tweaked/data/test/nsy1280' \
    --dataset='nt4096_ls64_nzf16_nzd32.pth' \
    --cutoff=30. --imageSize=4096 --latentSize=64 \
    --niter=1001 --cuda --ngpu=4 --nzd=16 --nzf=8 --rlr=0.00025164314945158394 --glr=0.0003 \
    --outf='./imgs_bb_ls64_nf8_nzd32/unic/config_27/ter/filtered/' \
    --workers=8 --nsy=1280 --batchSize=1024 \
    --actions='./action/actions_unic.txt' --strategy='./strategy/strategy_unic.txt' \
    --save_checkpoint=1000 --root_checkpoint='./network/bb_ls64_nf8_nzd32/unic/config_27/ter/filtered/'\
    --config='./config/unic-zyy16-zxy8-wgan-1.json'

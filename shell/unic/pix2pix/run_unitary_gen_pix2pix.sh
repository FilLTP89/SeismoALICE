#!/bin/bash
#SBATCH -J pix2pix
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:2
#SBATCH --mem=150GB
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --output=./output/unic/pix2pix/out_gen.txt
#SBATCH --error=./error/unic/pix2pix/error_gen.txt
#SBATCH --mail-type=FAIL
#SBATCH --export=NONE  

source load_conda_env_gpu.sh P100  
export PYTHONPATH="./src"

python ./src/aae/unic_test/pix2pix/aae_unitary_test_gen_pix2pix.py \
    --dataroot='./database/tweaked/data/test/nsy12800/' \
    --dataset='nt4096_ls128_nzf8_nzd32.pth' \
    --cutoff=30. --imageSize=4096 --latentSize=64 \
    --niter=1501 --cuda --nodes=1 \
    --local_rank=0 --ngpu=2 --ip_address=$ip1 --nzd=16 --nzf=8 \
    --rlr=0.00025164314945158394 --glr=0.006187098496095162 --manualSeed=42\
    --outf='./imgs/simple/wgan/bb/' \
    --workers=8 --nsy=12800 --batchSize=256 \
    --actions='./action/actions_unic.txt' --strategy='./strategy/strategy_unic.txt' \
    --save_checkpoint=500 \
    --root_checkpoint='./network/test/unic/pix2pix/'\
    --config='./config/unic/pix2pix/pix2pix.json' 

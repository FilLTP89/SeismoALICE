#!/bin/bash
#SBATCH -J alice
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:2
#SBATCH --mem=150GB
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --output=./output/simple/alice/out_disc.txt
#SBATCH --error=./error/simple/alice/error_disc.txt 
#SBATCH --mail-type=FAIL
#SBATCH --export=NONE  
source load_conda_env_gpu.sh P100  
export PYTHONPATH="./src"

python ./src/aae/simple_test/alice/aae_unitary_test_disc_alice.py\
    --dataroot='./database/tweaked/data/test/nsy12800/' \
    --dataset='nt4096_ls128_nzf8_nzd32.pth' \
    --cutoff=30. --imageSize=4096 --latentSize=64 \
    --niter=1501 --cuda --nodes=1 \
    --local_rank=0 --ngpu=2 --ip_address=$ip1 --nzd=16 --nzf=8 \
    --rlr=0.00025164314945158394 --glr=0.006187098496095162 --manualSeed=42\
    --outf='./imgs_bb_ls64_nf8_nzd32/unic/config_27/ter/unic/classic/zyy24/nsy1280/' \
    --workers=8 --nsy=12800 --batchSize=256 \
    --actions='./action/actions_unic.txt' --strategy='./strategy/strategy_unic.txt' \
    --save_checkpoint=500 \
    --root_checkpoint='./network/test/alice/bb/'\
    --config='./config/simple/alice/alice.json' 
#!/bin/bash
export PYTHONPATH="./src"

python3 ./src/aae_drive_bbfl.py --dataroot='./database/stead' --dataset='nt4096_ls128_nzf8_nzd32.pth'  --cutoff=1. --imageSize=4096 --latentSize=128  --niter=5000 --cuda --ngpu=1 --nzf=8 --rlr=0.0001 --glr=0.0001 --outf='./imgs' --workers=8 --nsy=100 --batchSize=100 --actions='./actions_fl.txt' --strategy='./strategy_fl.txt' 

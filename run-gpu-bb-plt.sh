#!/bin/bash
export PYTHONPATH="./src"

python3 ./src/aae_drive_bbfl.py --dataroot='database/mdof/' --dataset='nt4096_ls128_nzf8_nzd32.pth' --cutoff=1. --imageSize=4096 --latentSize=128  --niter=5000 --cuda --ngpu=1 --nzd=32 --rlr=0.0001 --glr=0.0001 --outf='./imgs' --workers=8 --nsy=100 --batchSize=10 --actions='./actions_bb_plt.txt' --strategy='./strategy_bb_plt.txt'
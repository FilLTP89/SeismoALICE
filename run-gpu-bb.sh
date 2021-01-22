#!/bin/bash
export PYTHONPATH="./src"

# STEAD
#python3 ./src/aae_drive_bbfl.py --dataroot='./database/stead' --dataset='nt4096_ls128_nzf8_nzd32.pth' --cutoff=1. --imageSize=4096 --latentSize=128  --niter=5000 --cuda --ngpu=1 --nzd=32 --rlr=0.0001 --glr=0.0001 --outf='./imgs' --workers=8 --nsy=5000 --batchSize=100 --actions='./actions_bb.txt' --strategy='./strategy_bb.txt' 

# MDOF
#python3 ./src/aae_drive_bbfl.py --dataroot='/workdir/invsem01/mdof/' --dataset='mdof' --cutoff=1. --imageSize=4096 --latentSize=128  --niter=7002 --cuda --ngpu=1 --nzd=32 --rlr=0.0001 --glr=0.0001 --outf='./imgs' --workers=8 --nsy=100 --batchSize=10 --actions='./actions_bb.txt' --strategy='./strategy_bb.txt' 

python3 ./src/aae_drive_bbfl.py --dataroot='./database/mdof' --dataset='nt4096_ls128_nzf8_nzd32.pth' --cutoff=1. --imageSize=4096 --latentSize=128  --niter=5002 --cuda --ngpu=1 --nzd=32 --rlr=0.0001 --glr=0.0001 --outf='./imgs' --workers=8 --nsy=100 --batchSize=10 --actions='./actions_bb.txt' --strategy='./strategy_bb.txt' 


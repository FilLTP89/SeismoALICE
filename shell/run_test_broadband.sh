#!/bin/bash
#SBATCH -J test
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --mem=120GB
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --output=oinfo.txt
#SBATCH --error=rinfo.txt 
#SBATCH --mail-type=FAIL
#SBATCH --export=NONE 
## 
source load_conda_env_gpu.sh P100  
## cd ${SLURM_SUBMIT_DIR}

 export PYTHONPATH="./src"

#python ./src/aae_drive_bbfl.py --dataroot='./database/stead' --dataset='nt4096_ls4_nf8_nzd32.pth' --cutoff=1. --imageSize=4096 --latentSize=4  --niter=5000 --cuda --ngpu=2 --nzd=32 --rlr=0.0001 --glr=0.0001 --outf='./imgs_bb_ls4_nf8_nzd32' --workers=8 --nsy=50 --batchSize=10 --actions='./actions_bb.txt' --strategy='./strategy_bb.txt' --save_checkpoint=2000 --config='./config/config_bb_ls4_nf8_ndz32.json'> log_bb.txt
python ./src/aae/aae_drive_unic_tweaked_multi_branch_broadband.py \
    --dataroot='./database/tweaked/data/test/nsy1280/' \
    --dataset='nt4096_ls64_nzf16_nzd32.pth' \
    --cutoff=30. --imageSize=4096 --latentSize=64 \
    --niter=3001 --cuda --nodes=1 \
    --local_rank=0 --ngpu=4 --ip_address=$ip1 --nzd=16 --nzf=8 \
    --rlr=0.00025164314945158394 --glr=0.006187098496095162 \
    --outf='./imgs_bb_ls64_nf8_nzd32/unic/config_27/ter/unic/classic/zyy24/nsy1280/' \
    --workers=8 --nsy=1280 --batchSize=256 \
    --actions='./action/actions_unic.txt' --strategy='./strategy/strategy_unic.txt' \
    --save_checkpoint=200 \
    --summary_dir='./runs_both/broadband/zyy16/back-test/nsy1280/test-dxy/dummy/test-deterministic/'\
    --root_checkpoint='./network/bb/zyy24/nsy1280/hack/dummy'\
    --config='./config/unic-zyy16-zxy8-2.json' 
#python -m torch.distributed.launch  --nnodes=2 --node=1 --master_addr $MASTER_ADDR --master_port 8390 ./src/aae/ddp/aae_unic_ddp.py --dataroot='./database/tweaked/data/test/nsy12800/' --dataset='nt4096_ls128_nzf8_nzd32.pth' --cutoff=30. --imageSize=4096 --latentSize=64 --niter=4001 --cuda --nodes=2 --local_rank=1 --ngpu=2 --ip_address=$ip1 --nzd=16 --nzf=8 --rlr=0.0020338064625882507 --glr=0.0202756041182898 --outf='./imgs_bb_ls64_nf8_nzd32/unic/config_27/ter/broadband/classic/zyy16/nsy12800/' --workers=8 --nsy=1280 --batchSize=256 --actions='./action/actions_unic.txt' --strategy='./strategy/strategy_unic.txt' --save_checkpoint=1000 --root_checkpoint='./network/bb_ls64_nf8_nzd32/unic/config_27/ter/broadband/classic/zyy16/nsy12800/' --config='./config/bb_ls64/test/ndz32/tentative_tweaked_unic_27_ter_test_1.json'


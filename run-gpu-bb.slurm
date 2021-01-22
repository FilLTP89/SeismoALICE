#!/bin/bash
#SBATCH -J run 
#SBATCH --ntasks=8
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=ro.txt 
#SBATCH --error=re.txt 
#SBATCH --partition=gpu

#chargement des modules
module purge
source $WORKDIR/load_conda_env_gpu.sh ruche


./run-gpu-bb.sh > output.mdof

#python aae_drive.py --dataroot="database/ann2bb/" --dataset="ann2bb.pth" --cutoff=1. --imageSize=4096 --latentSize=128  --niter=1 --cuda --ngpu=1 --nzd=32 --rlr=0.0001 --glr=0.0001 --outf='./imgs' --workers=8 --nsy=100 --batchSize=10 --actions='./actions_hb_cmp.txt' --strategy='./strategy_hb_cmp.txt' > log_hb.txt & 

#python aae_drive_plot.py --dataroot='database/stead' --dataset='nt4096_ls128_nzf8_nzd32.pth' --cutoff=1. --imageSize=4096 --latentSize=128  --niter=5000 --cuda --ngpu=1 --nzd=32 --rlr=0.0001 --glr=0.0001 --outf='./imgs' --workers=8 --nsy=5000 --batchSize=100 --actions='./actions_bb_plt.txt' --strategy='./strategy_bb_plt.txt' > log_bb_plot.txt 
#python aae_drive_plot.py --dataroot='./' --dataset='ns5000_nt4096_ls128_nzf8_nzd32.pth' --cutoff=1. --imageSize=4096 --latentSize=128  --niter=4000 --cuda --ngpu=3 --nzd=32 --rlr=0.0001 --glr=0.0001 --outf='./imgs' --workers=8 --nsy=1000 --batchSize=100  --actions='./actions_bb.txt' --strategy='./strategy_bb.txt' > log_bb.txt

#    python aae_drive.py --dataroot="database/ann2bb/" --dataset="ann2bb.pth" --cutoff=1. --imageSize=4096 --latentSize=128  --niter=1 --cuda --ngpu=1 --nzd=32 --rlr=0.0001 --glr=0.0001 --outf='./imgs' --workers=8 --nsy=100 --batchSize=10 --actions='./actions_hb_cmp.txt' --strategy='./strategy_hb_cmp.txt' > log_hb.txt & 

#    python aae_drive.py --dataroot='database/stead/' --dataset='nt4096_ls128_nzf8_nzd32.pth' --cutoff=1. --imageSize=4096 --latentSize=128  --niter=7000 --cuda --ngpu=1 --nzd=32 --rlr=0.0001 --glr=0.0001 --outf='./imgs' --workers=8 --nsy=100 --batchSize=10  --actions='./actions_hb_plt.txt' --strategy='./strategy_hb_plt.txt' > log_hb.txt

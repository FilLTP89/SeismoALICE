#!/bin/bash
#SBATCH -J stead2pth
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:2
#SBATCH --mem=80GB
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --output=outputJob-stead2pth.txt
#SBATCH --error=errorJob-stead2pth.txt 
#SBATCH --mail-type=FAIL
#SBATCH --export=NONE 
## 
#source load_conda_env_gpu.sh ruche  
## cd ${SLURM_SUBMIT_DIR}
 
export PYTHONPATH="./src"

mpirun --use-hwthread-cpus -np 12 python3 ./src/STEADextractorMPI.py --dataroot=$HOME/Data/Filippo/aeolus/STEAD/waveforms_11_13_19.hdf5 --dataset='stead' --cutoff=1. --signalSize=4096 --latentSize=4 --nzd=128 --workers=2 --nsy=10000 --batchSize=50

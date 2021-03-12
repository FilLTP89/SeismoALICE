#!/bin/bash
#SBATCH -J STEAD2pthMPI
#SBATCH --nodes=1
#SBATCH --ntasks=20
#SBATCH --time=01:00:00
#SBATCH --partition=cpu_short
#SBATCH --output=outputJob-STEAD2pthMPI.txt
#SBATCH --error=errorJob-STEAD2pthMPI.txt 
#SBATCH --mail-type=FAIL
#SBATCH --export=NONE 

source $HOME/load_conda_env_gpu.sh
cd ${SLURM_SUBMIT_DIR}
 
export PYTHONPATH="./src"
export STEADROOT="/gpfs/workdir/invsem03/STEAD/waveforms_11_13_19.hdf5 "
mpirun --use-hwthread-cpus -np 20 python3 ./src/STEADextractorMPI.py --dataroot=${STEADROOT} --dataset='stead' --cutoff=1. --signalSize=4096 --latentSize=4 --nzd=128 --workers=2 --nsy=10000 --batchSize=50

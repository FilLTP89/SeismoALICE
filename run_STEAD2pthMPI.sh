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

source $WORKDIR/load_conda_env_gpu.sh ruche
cd ${SLURM_SUBMIT_DIR}
 
export PYTHONPATH="./src"
export STEADROOT="/gpfs/workdir/jacquetg/STEAD/waveforms_11_13_19.hdf5"
mpirun -np 20 python ./src/STEADextractorMPI.py --dataroot=${STEADROOT} --dataset='stead' --cutoff=1. --signalSize=4096 --latentSize=4 --nzd=128 --nsy=1000 --batchSize=50

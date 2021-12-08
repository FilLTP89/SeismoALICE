#!/bin/bash
export PYTHONPATH="./src"
export STEADROOT="/home/kltpk89/Data/Filippo/aeolus/STEAD/waveforms_11_13_19.hdf5"
mpirun --use-hwthread-cpus -np 12 python3 ./src/STEADextractorMPI.py --dataroot=${STEADROOT} --dataset='stead' --cutoff=1. --signalSize=4096 --latentSize=64 --nzd=32 --nzf=32 --workers=2 --nsy=100000 

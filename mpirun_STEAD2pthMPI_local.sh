export PYTHONPATH="./src"
export STEADROOT="../STEAD/waveforms_11_13_19.hdf5"
mpirun --use-hwthread-cpus -np 20 python3 ./src/STEADextractorMPI.py --dataroot=${STEADROOT} --dataset='stead' --cutoff=1. --signalSize=4096 --latentSize=4 --nzd=128 --workers=2 --nsy=10000 --batchSize=50

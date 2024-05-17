#!/bin/bash -l

#PBS -P v19
#PBS -q normal
#PBS -l walltime=8:00:00
#PBS -l ncpus=48
#PBS -l mem=190GB
#PBS -l jobfs=400GB
#PBS -l wd
#PBS -l storage=gdata/xv83+gdata/gb6+gdata/v14+gdata/es60+scratch/es60+gdata/al33+gdata/cj50+gdata/dk92+gdata/fs38+gdata/ik11+gdata/oi10+gdata/p73+gdata/rr3+gdata/xp65
#PBS -j oe 
#PBS -M thomas.moore@csiro.au
#PBS -m abe

conda activate pangeo_bran2020_demo

python -u ../run_rechunker_loop.py > ./logs/$PBS_JOBID-megamem-rechunker_loop.log 2>&1




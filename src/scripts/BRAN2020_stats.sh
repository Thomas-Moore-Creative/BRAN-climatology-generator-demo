#!/bin/bash -l

#PBS -P v19
#PBS -q megamem
#PBS -l walltime=48:00:00
#PBS -l ncpus=48
#PBS -l mem=2990GB
#PBS -l jobfs=1400GB
#PBS -l wd
#PBS -l storage=gdata/xv83+gdata/gb6+gdata/v14+gdata/es60+scratch/es60+gdata/al33+gdata/cj50+gdata/dk92+gdata/fs38+gdata/ik11+gdata/oi10+gdata/p73+gdata/rr3+gdata/xp65
#PBS -j oe 
#PBS -M thomas.moore@csiro.au
#PBS -m abe

conda activate pangeo_bran2020_demo

python -u ../BRAN2020_stats.py > ./logs/$PBS_JOBID-megamem-3D-salt-stats.log 2>&1

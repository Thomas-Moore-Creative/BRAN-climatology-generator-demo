#!/bin/bash -l

#PBS -P es60
#PBS -q megamem
#PBS -l walltime=24:00:00
#PBS -l ncpus=48
#PBS -l mem=2990GB
#PBS -l jobfs=1400GB
#PBS -l wd
#PBS -l storage=gdata/gb6+gdata/v14+gdata/es60+scratch/es60+gdata/al33+gdata/cj50+gdata/dk92+gdata/fs38+gdata/ik11+gdata/oi10+gdata/p73+gdata/rr3+gdata/xp65+gdata/xv83
#PBS -j oe

conda activate pangeo_regionmask

python ../make_BRAN2020_clim.py
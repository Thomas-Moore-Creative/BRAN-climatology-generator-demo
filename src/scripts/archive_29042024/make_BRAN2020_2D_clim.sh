#!/bin/bash -l

#PBS -P es60
#PBS -q normalsr
#PBS -l walltime=12:00:00
#PBS -l ncpus=208
#PBS -l mem=500GB
#PBS -l jobfs=400GB
#PBS -l wd
#PBS -l storage=gdata/gb6+gdata/v14+gdata/es60+scratch/es60+gdata/al33+gdata/cj50+gdata/dk92+gdata/fs38+gdata/ik11+gdata/oi10+gdata/p73+gdata/rr3+gdata/xp65
#PBS -j oe

conda activate pangeo_bran2020_demo

python ../make_BRAN2020_2D_clim.py

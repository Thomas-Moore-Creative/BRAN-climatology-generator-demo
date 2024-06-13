#!/bin/bash
  
#PBS -l ncpus=1
#PBS -l mem=2GB
#PBS -l jobfs=100GB
#PBS -q copyq
#PBS -P es60
#PBS -l walltime=06:00:00
#PBS -l wd
#PBS -l storage=gdata/xv83+gdata/gb6+gdata/v14+gdata/es60+scratch/es60+gdata/al33+gdata/cj50+gdata/dk92+gdata/fs38+gdata/ik11+gdata/oi10+gdata/p73+gdata/rr3+gdata/xp65
#PBS -j oe 
#PBS -M thomas.moore@csiro.au
#PBS -m abe
  
#rsync -avPs /g/data/es60/users/thomas_moore/clim_demo_results/results_backup/. /g/data/xv83/users/tm4888/PROJECTS/BRAN2020/results_backup/. > ./logs/$PBS_JOBID-data-move.log 2>&1
rsync -avPs /g/data/xv83/users/tm4888/PROJECTS/BRAN2020/BRAN2020_climatology/. /g/data/es60/users/thomas_moore/clim_demo_results/daily/bran2020_GISready_results/. > ./logs/$PBS_JOBID-data-move.log 2>&1
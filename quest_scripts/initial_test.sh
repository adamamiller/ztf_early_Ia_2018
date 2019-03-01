#!/bin/bash
#MSUB -A p30796
#MSUB -q normal
#MSUB -l walltime=24:00:00
#MSUB -M my_email_address
#MSUB -j oe
#MSUB -N ztf_2018_snIa
#MSUB -l mem=120gb
#MSUB -l nodes=1:ppn=28
#MSUB -l partition=quest6

# add a project directory to your PATH (if needed)
export PATH=$PATH:/projects/p30796/tools/

# load modules you need to use
module load python/anaconda
source activate emcee3

# Run the actual commands to be executed
cd /home/aam3503/software/adamamiller_git_clones/ztf_early_Ia_2018/playground
python dummy_run_script.py
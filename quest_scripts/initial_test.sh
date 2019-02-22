#!/bin/bash
#MSUB -A p30796
#MSUB -q short
#MSUB -l walltime=04:00:00
#MSUB -M my_email_address
#MSUB -j oe
#MSUB -N ztf_2018_snIa
#MSUB -l nodes=1:ppn=4

# add a project directory to your PATH (if needed)
export PATH=$PATH:/projects/p30796/tools/

# load modules you need to use
module load python/anaconda
source activate emcee3

# Set your working directory
cd $PBS_O_WORKDIR/software/adamamiller_git_clones/ztf_early_Ia_2018/playground

# Another command you actually want to execute, if needed:
python dummy_run_script.py
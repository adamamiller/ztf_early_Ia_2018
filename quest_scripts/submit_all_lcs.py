'''
Short script to run on quest to create and submit a request for 
forced photometry for every source in the 2018 sample
'''

import pandas as pd
import numpy as np
import glob
import subprocess

info_path="/projects/p30796/ZTF/early_Ia/2018/info/"

source_files = glob.glob(info_path+'force_phot*.fits')
for source_file in source_files:
    source = source_file.split('/')[-1].split('_')[2]
    
    with open('{}_force_phot.sh'.format(source), 'w') as fw:
        print('''#!/bin/bash
#MSUB -A p30796
#MSUB -q short
#MSUB -l walltime=04:00:00
#MSUB -M my_email_address
#MSUB -j oe
''',file=fw)
        print('#MSUB -N sys_{}'.format(source),file=fw)
        print('''#MSUB -l mem=90gb
#MSUB -l nodes=1:ppn=28
#MSUB -l partition=quest8
                 
# add a project directory to your PATH (if needed)
export PATH=$PATH:/projects/p30796/tools/

# load modules you need to use
module load python/anaconda
source activate emcee3

# Run the actual commands to be executed
cd /home/aam3503/software/adamamiller_git_clones/ForcePhotZTF
python parallel_force_lc.py {}
'''.format(source),file=fw)

    subprocess.call(['chmod', 'ugo+x', '{}_force_phot.sh'.format(source)])
    subprocess.call(['msub', '{}_force_phot.sh'.format(source)])
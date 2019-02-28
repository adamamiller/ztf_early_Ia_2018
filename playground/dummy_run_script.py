import fit_ztf_lc
import pandas as pd

import os
os.environ["OMP_NUM_THREADS"] = "1"

sn = 'ZTF18aaxsioa'
data_path = "/projects/p30796/ZTF/2018/early_Ia/"
lc_df = pd.read_csv(data_path + "forced_lightcurves/{}.csv".format(sn))
salt_df = pd.read_csv(data_path + "MB_SALT_020419.csv")

t0 = float(salt_df['t0'][salt_df['sn'] == sn].values) + 2400000.5
z = float(salt_df['z'][salt_df['sn'] == sn].values)

fit_ztf_lc.fit_lc(lc_df, t0=t0, z=z, t_fl=18, 
                  mcmc_h5_file=data_path + "dummy.h5")

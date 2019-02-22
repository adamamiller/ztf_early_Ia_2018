import fit_ztf_lc
import pandas as pd

sn = 'ZTF18aaxsioa'
data_path = "../../Marshall_data/growth_marshall_lcs/"
lc_df = pd.read_csv(data_path + "forced_lightcurves/{}.csv".format(sn))
salt_df = pd.read_csv(data_path + "MB_SALT_020419.csv")

t0 = float(salt_df['t0'][salt_df['sn'] == sn].values) + 2400000.5
z = float(salt_df['z'][salt_df['sn'] == sn].values)

fit_ztf_lc.fit_lc(lc_df, t0=0, z=0, t_fl=18)
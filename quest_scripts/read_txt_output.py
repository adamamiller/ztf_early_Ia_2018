'''
Short script to read results
'''

import numpy as np
import pandas as pd
import glob
import emcee

info_path="/projects/p30796/ZTF/early_Ia/forced_lightcurves/sample_lc/"
salt_df = pd.read_csv(info_path + "../../Nobs_cut_salt2_spec_subtype.csv")

output_files = glob.glob('ZTF18*_gr_exp50percent.sh.o450*')

name_arr = np.empty(len(output_files)).astype('S12')
conv_arr = np.zeros(len(output_files)).astype(bool)
t0_arr = np.empty(len(output_files)).astype('float')
alpha_g_arr = np.empty(len(output_files)).astype('float')
alpha_r_arr = np.empty(len(output_files)).astype('float')

for filenum, ofile in enumerate(output_files): 
    ztf_name = ofile.split('_gr_')[0]
    name_arr[filenum] = ztf_name
    with open(ofile) as f:
        ll = f.readlines()
    if len(ll) == 118:
        print('For {} model ran 1200000 steps'.format(ztf_name))
        if np.all(np.array(ll[115].split('[')[1].split('\n')[0].split(), dtype=float) < 200) and np.all(np.array(ll[116].split(']')[0].split(), dtype=float) < 200):
            print('\t model also converged')
            conv_arr[filenum] = True
    if 50 < len(ll) < 118:
        nsteps = int(ll[-3].split(' ')[1])
        print('For {} model ran {} steps'.format(ztf_name, nsteps))
        conv_arr[filenum] = True
    if len(ll) > 50:
        h5_file = info_path + '{}_emcee.h5'.format(ztf_name)
        reader = emcee.backends.HDFBackend(h5_file)
        tau = reader.get_autocorr_time(tol=0)
        burnin = int(5*np.max(tau))
        samples = reader.get_chain(discard=burnin, flat=True)
        t0 = np.median(samples[:,0])
        alpha_g = np.median(samples[:,3])
        alpha_r = np.median(samples[:,6])
        t0_arr[filenum] = t0
        alpha_g_arr[filenum] = alpha_g
        alpha_r_arr[filenum] = alpha_r    

res_df = pd.DataFrame(data=name_arr, columns=['ztf_name'])
res_df['t0'] = t0_arr
res_df['alpha_g'] = alpha_g_arr
res_df['alpha_r'] = alpha_r_arr
res_df['conv_arr'] = conv_arr.astype(int)
res_df.to_csv('intermediate_results.csv', index=False)
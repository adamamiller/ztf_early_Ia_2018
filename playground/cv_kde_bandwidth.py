import numpy as np
import pandas as pd
import time
import sys
import emcee

from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity


def get_all_bandwidths(h5_file, 
                       thin_by=100, 
                       data_path = '',
                       n_cores=8,
                       prior='uninformative',
                       **kwargs):
    '''optimal bandwidth for marginilized KDEs
    
       warning - lots of hard coding'''
    sn = h5_file.split('/')[-1].split('_')[0]
    reader = emcee.backends.HDFBackend(h5_file)
    nsteps = thin_by*np.shape(reader.get_chain())[0]
    tau = reader.get_autocorr_time(tol=0)
    burnin = int(5*np.max(tau))
    samples = reader.get_chain(discard=burnin, 
                               thin=np.max([int(np.max(tau)), 1]), 
                               flat=True)
    lnpost = reader.get_log_prob(discard=burnin, 
                                 thin=np.max([int(np.max(tau)), 1]), 
                                 flat=True)
    t_fl = samples[:,0]

    time_bw = opt_bandwidth(t_fl, 
                            log_min_grid=-2.7,
                            log_max_grid=-0.3,
                            n_jobs=n_cores)
    if prior == 'uninformative':
        alpha_g = samples[:,2]
        alpha_r = samples[:,4]
        delta_df = pd.read_hdf(data_path + '{}_{}_deltas.h5'.format(sn, prior))
        delta = delta_df.delta.values


        alpha_g_bw = opt_bandwidth(alpha_g, 
                                   log_min_grid=-2.7,
                                   log_max_grid=-0.3,
                                   n_jobs=n_cores)
        alpha_r_bw = opt_bandwidth(alpha_r, 
                                   log_min_grid=-2.7,
                                   log_max_grid=-0.3,
                                   n_jobs=n_cores)
        delta_bw = opt_bandwidth(delta, 
                                 log_min_grid=-2.7,
                                 log_max_grid=-0.3,
                                 n_jobs=n_cores)
    
    
    with open(data_path + '{}_{}_bandwidth.txt'.format(sn, prior), 'w') as fw:
        print('{} = bw for time_fl'.format(time_bw),file=fw)
        if prior == 'uninformative':
            print('{} = bw for alpha_g'.format(alpha_g_bw),file=fw)
            print('{} = bw for alpha_r'.format(alpha_r_bw),file=fw)
            print('{} = bw for delta'.format(delta_bw),file=fw)
    
    return

def opt_bandwidth(marg_samples, 
                  log_min_grid=-2.5,
                  log_max_grid=0,
                  grid_points=15,
                  n_cv=3,
                  n_jobs=-1):
    '''determine the optimal KDE bandwidth via gridsearch CV'''
    
    params = {'bandwidth': np.logspace(log_min_grid, 
                                       log_max_grid, 
                                       grid_points)}
    grid_cv = GridSearchCV(KernelDensity(rtol=1e-4), 
                           params, cv=n_cv, n_jobs=n_jobs)
    if len(np.shape(marg_samples)) == 1:
        X_marg = np.reshape(marg_samples, (len(marg_samples), 1))
    else: 
        X_marg = marg_samples
    grid_cv.fit(X_marg)

    return grid_cv.best_estimator_.bandwidth
    
if __name__== "__main__":
    ztf_name = str(sys.argv[1])
    n_cores = 8
    prior = 'uninformative'
    if len(sys.argv) > 2:
        n_cores = int(sys.argv[2])
    if len(sys.argv) > 3:
        prior = 'delta2'
    
    data_path = "/projects/p30796/ZTF/early_Ia/forced_lightcurves/sample_lc_v2/big_unc/"
    
    if prior == 'uninformative':
        backend_filename = data_path + "/{}_emcee_40_varchange.h5".format(ztf_name)
        get_all_bandwidths(backend_filename, data_path=data_path, n_cores=n_cores)
    elif prior == 'delta2':
        backend_filename = data_path + "/{}_emcee_40_tsquared.h5".format(ztf_name)
        get_all_bandwidths(backend_filename, data_path=data_path, 
                           n_cores=n_cores, prior=prior)
        
import numpy as np
import pandas as pd
import time
import sys
import emcee

from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity


def get_all_bandwidths(h5_file, 
                       thin_by=250, 
                       data_path = '',
                       **kwargs):
    '''optimal bandwidth for marginilized KDEs
    
       warning - lots of hard coding'''
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
    alpha_g = samples[:,2]
    alpha_r = samples[:,4]
    delta_alpha = alpha_r - alpha_g

    time_bw = opt_bandwidth(t_fl, 
                            log_min_grid=-1.1,
                            log_max_grid=0.5,
                            n_jobs=4)
    alpha_g_bw = opt_bandwidth(alpha_g, 
                               n_jobs=4)
    alpha_r_bw = opt_bandwidth(alpha_r, 
                               n_jobs=4)
    delta_alpha_bw = opt_bandwidth(delta_alpha, 
                                   log_min_grid=-1.5,
                                   log_max_grid=0.0,
                                   n_jobs=4)
    
    sn = h5_file.split('/')[-1].split('_')[0]
    with open(data_path + '{}_bandwidth.txt'.format(sn)) as fw:
        print('{} = bw for time_fl'.format(time_bw))
        print('{} = bw for alpha_g'.format(alpha_g_bw))
        print('{} = bw for alpha_r'.format(alpha_r_bw))
        print('{} = bw for delta_alpha'.format(delta_alpha_bw))
    
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
    X_marg = np.reshape(marg_samples, (len(marg_samples), 1))
    grid_cv.fit(X_marg)

    return grid_cv.best_estimator_.bandwidth
    
if __name__== "__main__":
    ztf_name = str(sys.argv[1])
    data_path = "/projects/p30796/ZTF/early_Ia/forced_lightcurves/sample_lc_v2/"
    backend_filename = data_path + "/{}_emcee.h5".format(ztf_name)

    get_all_bandwidths(backend_filename, data_path)
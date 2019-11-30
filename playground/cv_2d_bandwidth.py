import numpy as np
import pandas as pd
import time
import sys
import emcee

from scipy.interpolate import RectBivariateSpline

from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity

from cv_kde_bandwidth import opt_bandwidth

def get_2d_bandwidth(h5_file, 
                     thin_by=100, 
                     data_path = '',
                     n_cores=8,
                     prior='uninformed',
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

    X_alphas = samples[:,(2,4)]

    alphas_bw = opt_bandwidth(X_alphas, 
                              log_min_grid=-2.7,
                              log_max_grid=0,
                              n_jobs=n_cores)
    
    kde = KernelDensity(rtol=1e-4, bandwidth=alphas_bw)

    kde.fit(X_alphas)
    
    xx, yy = np.mgrid[0:7.5:0.0075, 
                      0:7.5:0.0075]
    xy_sample = np.vstack([yy.ravel(), xx.ravel()]).T
    z = kde.score_samples(xy_sample)
    zz = np.reshape(z, xx.shape)
    
    if prior == 'uninformed':
        cdf_g = np.cumsum(np.sum(np.exp(zz)*10**(xx+yy), axis=0)) # sum along g
        rand_uni = np.random.uniform(size=len(samples))
        alpha_g = np.interp(rand_uni, cdf_g/cdf_g[-1], xx[:,0])
        
        f = RectBivariateSpline(xx[:,0], yy[0], np.exp(zz)*10**(xx+yy))
        
        # loop to get all the alpha_r
        alpha_r = np.zeros_like(alpha_g)
        for al_num, al in enumerate(alpha_g):
            pdf_r = f(yy[0], al)
            cdf_r = np.cumsum(pdf_r[:,0])
            rand_num = np.random.uniform()
            alpha_r[al_num] = np.interp(rand_num, cdf_r/cdf_r[-1], yy[0])
        delta = alpha_r - alpha_g
    
    delta_df = pd.DataFrame(delta, columns=['delta'])
    delta_df.to_hdf(data_path + '{}_{}_deltas.h5'.format(sn, prior),'d1')
        
    return

if __name__== "__main__":
    ztf_name = str(sys.argv[1])
    n_cores = 8
    prior = 'uninformed'
    if len(sys.argv) > 2:
        n_cores = int(sys.argv[2])
    
    data_path = "/projects/p30796/ZTF/early_Ia/forced_lightcurves/sample_lc_v2/big_unc/"
    
    if prior == 'uninformed':
        backend_filename = data_path + "/{}_emcee_40_varchange.h5".format(ztf_name)
        get_2d_bandwidth(backend_filename, data_path, n_cores=n_cores)

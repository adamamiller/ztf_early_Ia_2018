import numpy as np
import pandas as pd
import emcee
import corner
from scipy.optimize import minimize
from multiprocessing import Pool, cpu_count
import time
import sys

def f_t(times, amplitude=25, t_0=0, alpha_r=2):
    '''Calculate the flux of an exponential rise explosion at time t
    
    Parameters
    ----------
    times : array-like
        The times, in units of days, at which the flux is to be calculated
    
    amplitude : float (default=25)
        The normalization amplitude for the flux
    
    t_0 : float (default=0)
        The time of explosion 

    alpha_r : float (default=2)
        Power-law index for the rising portion of the light curve
     '''
    return amplitude * (times - t_0)**alpha_r

# define the likelihood and associated functions
def lnlike_simple(theta, f, t, f_err):
    t_0, a, a_prime, alpha_r = theta
    
    pre_exp = np.logical_not(t > t_0)
    model = np.empty_like(f)
    model[pre_exp] = a
    
    time_term = (t[~pre_exp] - t_0)
    model[~pre_exp] = a + a_prime * (time_term)**alpha_r
    
    ln_l = -0.5*np.sum((f - model)**2 / (f_err**2))
    return ln_l

def nll_simple(theta, f, t, f_err):
    return -1*lnlike_simple(theta, f, t, f_err)

#Define priors on parameters  
def lnprior_simple(theta):
    t_0, a, a_prime, alpha_r = theta
    if (-1e8 < t_0 < 1e8 and 0 < alpha_r < 1e8 and 
        -1e8 < a < 1e8 and 
        0 < a_prime < 1e8):
        return 0.0
    return -np.inf

def lnposterior_simple(theta, f, t, f_err):
    lnp = lnprior_simple(theta)
    lnl = lnlike_simple(theta, f, t, f_err)
    if not np.isfinite(lnl):
        return -np.inf
    if not np.isfinite(lnp):
        return -np.inf
    return lnl + lnp

#############################################
## Define likelihood for multi-filter data ##
#############################################

def multifilter_lnlike_simple(theta, f, t, f_err, filt_arr):
    
    if len(theta) % 3 != 1:
        raise RuntimeError('The correct number of parameters were not included')
    
    ln_l = 0
    for filt_num, filt in enumerate(np.unique(filt_arr)):
        theta_filt = np.append(theta[0], theta[1+3*filt_num:4+3*filt_num])
        filt_obs = np.where(filt_arr == filt)
        f_filt = f[filt_obs]
        t_filt = t[filt_obs]
        f_err_filt = f_err[filt_obs]
        ln_l += lnlike_simple(theta_filt, f_filt, t_filt, f_err_filt)
    
    return ln_l

def multifilter_nll_simple(theta, f, t, f_err, filt_arr):
    return -1*multifilter_lnlike_simple(theta, f, t, f_err, filt_arr)

def multifilter_lnprior_simple(theta, filt_arr):
    
    if len(theta) % 3 != 1:
        raise RuntimeError('The correct number of parameters were not included')
    
    ln_p = 0
    for filt_num, filt in enumerate(np.unique(filt_arr)):
        theta_filt = np.append(theta[0], theta[1+3*filt_num:4+3*filt_num])
        ln_p += lnprior_simple(theta_filt)
    return ln_p

def multifilter_lnposterior_simple(theta, f, t, f_err, filt_arr):
    lnp = multifilter_lnprior_simple(theta, filt_arr)
    lnl = multifilter_lnlike_simple(theta, f, t, f_err, filt_arr)
    if not np.isfinite(lnl):
        return -np.inf
    if not np.isfinite(lnp):
        return -np.inf
    return lnl + lnp
    
def fit_lc(lc_df, t0=0, z=0, t_fl=17, 
           mcmc_h5_file="ZTF_SN.h5",
           max_samples=int(2e6),
           nwalkers=100,
           g_rel_flux_cutoff = 0.5,
           ncores=None,
           use_emcee_backend=True,
           thin_by=1,
           g_max=1,
           r_max=1):
    '''Perform an MCMC fit to the light curve'''
    t_mcmc_start = time.time()
    
    if ncores == None:
        ncores = cpu_count() - 1
    
    obs = np.where( (lc_df['programid'] <= 2.0) & 
                    (lc_df['offset'] > -999)  
                  )

    g_obs = np.where(lc_df.iloc[obs]['filter'] == b'g')
    r_obs = np.where(lc_df.iloc[obs]['filter'] == b'r')

    time_rf = (lc_df['jdobs'].iloc[obs].values - t0)/(1+z)
    flux = lc_df['Fratio'].iloc[obs].values
    if g_max == 1:
        g_max = np.max(lc_df['Fratio'].iloc[obs[0][g_obs]].values)
    if r_max == 1:
        r_max = np.max(lc_df['Fratio'].iloc[obs[0][r_obs]].values)
    flux[g_obs] = flux[g_obs]/g_max
    flux[r_obs] = flux[r_obs]/r_max
    flux_unc = lc_df['Fratio_unc'].iloc[obs].values
    flux_unc[g_obs] = flux_unc[g_obs]/g_max
    flux_unc[r_obs] = flux_unc[r_obs]/r_max
    filt_arr = lc_df['filter'].iloc[obs].values

    t_fl = 18

    guess_0 = [-t_fl, 
               0, 6e-3, 2,
               0, 6e-3, 2
              ]    

    half_max_g = np.where((flux[g_obs] < g_rel_flux_cutoff) & (time_rf[g_obs] < 0))
    early_obs = np.where(time_rf <= time_rf[g_obs][np.max(half_max_g[0])])

    f_data = flux[early_obs]
    t_data = time_rf[early_obs]
    f_unc_data = flux_unc[early_obs]
    filt_data = filt_arr[early_obs]
    
    ml_res = minimize(multifilter_nll_simple, guess_0, method='Powell', # Powell method does not need derivatives
                      args=(f_data, t_data, f_unc_data, filt_data))
    ml_guess = ml_res.x
    
    ndim = len(ml_guess)
    nfac = [1e-2]*ndim

    #initial position of walkers
    pos = [ml_guess*(1 + nfac*np.random.randn(ndim)) for i in range(nwalkers)]

    with Pool(ncores) as pool:
        if use_emcee_backend:
            # file to save samples
            filename = mcmc_h5_file
            backend = emcee.backends.HDFBackend(filename)
            backend.reset(nwalkers, ndim)        

            sampler = emcee.EnsembleSampler(nwalkers, ndim, 
                                            multifilter_lnposterior_simple, 
                                            args=(f_data, t_data, 
                                                  f_unc_data, filt_data),
                                            pool=pool, backend=backend)
        else:
            sampler = emcee.EnsembleSampler(nwalkers, ndim, 
                                            multifilter_lnposterior_simple, 
                                            args=(f_data, t_data, 
                                                  f_unc_data, filt_data),
                                            pool=pool)
            
        max_samples = max_samples

        old_tau = np.inf
        for sample in sampler.sample(pos, 
                                     iterations=max_samples, 
                                     thin_by=thin_by, progress=False):
            if ((sampler.iteration <= int(1e3/thin_by)) and 
                 sampler.iteration % int(250/thin_by)):
                continue
            elif ((int(1e3/thin_by) < sampler.iteration <= int(1e4/thin_by)) 
                  and sampler.iteration % int(1e3/thin_by)):
                continue
            elif ((int(1e4/thin_by) < sampler.iteration <= int(1e5/thin_by)) 
                  and sampler.iteration % int(1e4/thin_by)):
                continue
            elif ((int(1e5/thin_by) < sampler.iteration) and 
                  sampler.iteration % int(2e4/thin_by)):
                continue
    
            tstart = time.time()
            tau = sampler.get_autocorr_time(tol=0)
            tend = time.time()
            steps_so_far = sampler.iteration
            print('''After {:d} steps, 
    autocorrelation takes {:.3f} s ({} total FFTs)                
    acceptance fraction = {:.4f}, and
    tau = {}'''.format(steps_so_far, 
                       tend-tstart, nwalkers*ndim,
                       np.mean(sampler.acceptance_fraction), 
                       tau))

            # Check convergence
            converged = np.all(tau * 100 < sampler.iteration)
            converged &= np.all(np.abs(old_tau - tau) / tau < 0.01)
            if converged:
                break
            old_tau = tau


    print("Ran {} steps; final tau= {}".format(steps_so_far*thin_by, tau))
    t_mcmc_end = time.time()
    print("All in = {:.2f} s on {} cores".format(t_mcmc_end - t_mcmc_start, 
                                                 ncores))

if __name__== "__main__":
    ztf_name = str(sys.argv[1])
    ncores = 27
    nsteps = int(1e6)
    thin_by = int(1)
    data_path = "/projects/p30796/ZTF/early_Ia/forced_lightcurves/sample_lc_v2/"
    backend_filename = data_path + "/{}_emcee.h5".format(ztf_name)
    use_emcee_backend = True
    
    if len(sys.argv) > 2:
        ncores = int(sys.argv[2])
    if len(sys.argv) > 3:
        nsteps = int(sys.argv[3])
    if len(sys.argv) > 4:
        thin_by = int(sys.argv[4])
    if len(sys.argv) > 5:
        backend_filename = str(sys.argv[5])
    if len(sys.argv) > 6:
        use_emcee_backend = False  

    lc_df = pd.read_hdf(data_path + "/{}_force_phot.h5".format(ztf_name))
    salt_df = pd.read_csv(data_path + "../../Nobs_cut_salt2_spec_subtype.csv")

    t0 = float(salt_df['t0_g_adopted'][salt_df['name'] == ztf_name].values)
    z = float(salt_df['z_adopt'][salt_df['name'] == ztf_name].values)
    g_max = float(salt_df['fratio_gmax_2adam'][salt_df['name'] == ztf_name].values)
    r_max = float(salt_df['fratio_rmax_2adam'][salt_df['name'] == ztf_name].values)
    
    fit_lc(lc_df, 
           t0=t0, z=z, 
           mcmc_h5_file=backend_filename, 
           max_samples=nsteps, 
           ncores=ncores,
           use_emcee_backend=use_emcee_backend,
           thin_by=thin_by,
           g_max=g_max,
           r_max=r_max)
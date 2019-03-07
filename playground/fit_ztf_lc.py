import numpy as np
import pandas as pd
import emcee
import corner
from scipy.optimize import minimize
from multiprocessing import Pool

def calc_t_p(t_b, alpha_d, alpha_r, s):
    '''Calculate the time to peak based on model parameters
    
    This parameterization of a SN light curve rise time is defined by 
    the model given in Eqn 8 in Zheng & Filippenko (2017), ApJL, 838, L4
        
    Parameters
    ----------
    t_b : float
        Break time for the broken power-law model
    
    alpha_d : float
        Power-law index for the declining portion of the light curve
    
    alpha_r : float
        Power-law index for the rising portion of the light curve
    
    s : float
        Model smoothing parameter for transition from rise to decline
    '''
    return t_b*(-1/(1 - 2*alpha_d/alpha_r))**(1/(s*alpha_d))

def f_t(times, amp=25, t_b=20, t_0=0, alpha_r=2, alpha_d=2, s=2):
    '''Calculate the flux of a model SN Ia at time t
    
    This parameterization of a SN light curve uses the model defined in Eqn 4 
    in Zheng & Filippenko (2017), ApJL, 838, L4.
    
    The default parameters yield a SN that peaks at m = 15 mag when adopting a 
    zeropoint = 25 (i.e. m = 25 - 2.5*log10[flux]).
    
    Parameters
    ----------
    times : array-like
        The times, in units of days, at which the flux is to be calculated
    
    amp : float (default=25)
        The normalization amplitude for the flux
    
    t_b : float (default=20)
        Break time for the broken power-law model

    t_0 : float (default=0)
        The time of explosion 

    alpha_r : float (default=2)
        Power-law index for the rising portion of the light curve

    alpha_d : float (default=2)
        Power-law index for the declining portion of the light curve    
    
    s : float (default=2)
        Model smoothing parameter for transition from rise to decline
    
     '''
    time_term = (times - t_0)/t_b
    
    return amp * (time_term)**alpha_r * (1 + (time_term)**(s*alpha_d))**(-2/s)
    
# create likelihood, prior, and posterior functions

def lnlikelihood(theta, f, t, f_err):
    t_0, a, a_prime, t_b, alpha_r, alpha_d, s, sig_0 = theta

    pre_exp = np.logical_not(t > t_0)
    model = np.empty_like(f)
    model[pre_exp] = a
    
    time_term = (t[~pre_exp] - t_0)/t_b
    model[~pre_exp] = a + a_prime * (time_term)**alpha_r * (1 + (time_term)**(s*alpha_d))**(-2/s)
    
    ln_l = np.sum(np.log(1. / np.sqrt(2*np.pi * (sig_0**2 + f_err**2))) - 
                 ((f - model)**2 / (2 * (sig_0**2 + f_err**2))))
    return ln_l

def lnlikelihood_no_sig0(theta, f, t, f_err):
    t_0, a, a_prime, t_b, alpha_r, alpha_d, s = theta

    pre_exp = np.logical_not(t > t_0)
    model = np.empty_like(f)
    model[pre_exp] = a
    
    time_term = (t[~pre_exp] - t_0)/t_b
    model[~pre_exp] = a + a_prime * (time_term)**alpha_r * (1 + (time_term)**(s*alpha_d))**(-2/s)
    
    chi2_term = -1/2*np.sum(((f - model)/f_err)**2)
    error_term = np.sum(np.log(1/np.sqrt(2*np.pi*f_err**2)))
    ln_l = chi2_term + error_term
    
    return ln_l

def nll(theta, f, t, f_err):
    return -1*lnlikelihood_no_sig0(theta, f, t, f_err)

def lnprior(theta):
    t_0, a, a_prime, t_b, alpha_r, alpha_d, s, sig_0 = theta
    if (-1e8 < t_0 < 1e8 and 0 < alpha_r < 1e8 and 
        0 < alpha_d < 1e8 and 0 < sig_0 < 1e8 and 
        -1e8 < a < 1e8 and  0 < t_b < 1e8 and 
        0 < s < 1e8 and 0 < a_prime < 1e8):
        return 0.0
    return -np.inf

def lnprior_no_sig0(theta):
    t_0, a, a_prime, t_b, alpha_r, alpha_d, s = theta
    if (-1e8 < t_0 < 1e8 and 0 < alpha_r < 1e8 and 
        0 < alpha_d < 1e8 and
        -1e8 < a < 1e8 and  0 < t_b < 1e8 and 
        0 < s < 1e8 and 0 < a_prime < 1e8):
        return 0.0
    return -np.inf

def lnposterior(theta, f, t, f_err):
    lnp = lnprior(theta)
    lnl = lnlikelihood(theta, f, t, f_err)
    if not np.isfinite(lnl):
        return -np.inf
    if not np.isfinite(lnp):
        return -np.inf
    return lnl + lnp

def lnposterior_no_sig0(theta, f, t, f_err):
    lnp = lnprior_no_sig0(theta)
    lnl = lnlikelihood_no_sig0(theta, f, t, f_err)
    if not np.isfinite(lnl):
        return -np.inf
    if not np.isfinite(lnp):
        return -np.inf
    return lnl + lnp

def multifilter_lnlikelihood(theta, f, t, f_err, filt_arr):
    
    if len(theta) % 7 != 1:
        raise RuntimeError('The correct number of parameters were not included')
    
    ln_l = 0
    for filt_num, filt in enumerate(np.unique(filt_arr)):
        theta_filt = np.append(theta[0], theta[1+7*filt_num:8+7*filt_num])
        filt_obs = np.where(filt_arr == filt)
        f_filt = f[filt_obs]
        t_filt = t[filt_obs]
        f_err_filt = f_err[filt_obs]
        ln_l += lnlikelihood(theta_filt, f_filt, t_filt, f_err_filt)
    
    return ln_l

def multifilter_lnlikelihood_no_sig0(theta, f, t, f_err, filt_arr):
    
    if len(theta) % 6 != 1:
        raise RuntimeError('The correct number of parameters were not included')
    
    ln_l = 0
    for filt_num, filt in enumerate(np.unique(filt_arr)):
        theta_filt = np.append(theta[0], theta[1+6*filt_num:7+6*filt_num])
        filt_obs = np.where(filt_arr == filt)
        f_filt = f[filt_obs]
        t_filt = t[filt_obs]
        f_err_filt = f_err[filt_obs]
        ln_l += lnlikelihood_no_sig0(theta_filt, f_filt, t_filt, f_err_filt)
    return ln_l

def multifilter_nll(theta, f, t, f_err, filt_arr):
    return -1*multifilter_lnlikelihood_no_sig0(theta, f, t, f_err, filt_arr)

def multifilter_prior(theta, filt_arr):
    
    if len(theta) % 7 != 1:
        raise RuntimeError('The correct number of parameters were not included')
    
    ln_p = 0
    for filt_num, filt in enumerate(np.unique(filt_arr)):
        theta_filt = np.append(theta[0], theta[1+7*filt_num:8+7*filt_num])
        ln_p += lnprior(theta_filt)
    return ln_p

def multifilter_prior_no_sig0(theta, filt_arr):
    
    if len(theta) % 6 != 1:
        raise RuntimeError('The correct number of parameters were not included')
    
    ln_p = 0
    for filt_num, filt in enumerate(np.unique(filt_arr)):
        theta_filt = np.append(theta[0], theta[1+6*filt_num:7+6*filt_num])
        ln_p += lnprior_no_sig0(theta_filt)
    return ln_p

def multifilter_lnposterior(theta, f, t, f_err, filt_arr):
    lnp = multifilter_prior(theta, filt_arr)
    lnl = multifilter_lnlikelihood(theta, f, t, f_err, filt_arr)
    if not np.isfinite(lnl):
        return -np.inf
    if not np.isfinite(lnp):
        return -np.inf
    return lnl + lnp

def multifilter_lnposterior_no_sig0(theta, f, t, f_err, filt_arr):
    lnp = multifilter_prior_no_sig0(theta, filt_arr)
    lnl = multifilter_lnlikelihood_no_sig0(theta, f, t, f_err, filt_arr)
    if not np.isfinite(lnl):
        return -np.inf
    if not np.isfinite(lnp):
        return -np.inf
    return lnl + lnp

def fit_lc(lc_df, t0=0, z=0, t_fl=18, 
           mcmc_h5_file="ZTF_SN.h5",
           max_samples=int(2e6),
           nwalkers=2000):
    '''Perform an MCMC fit to the light curve'''
    
    obs = np.where((lc_df['programid'] == 2.0) & 
                   (lc_df['nbaseline'] > 30))
    
    
    time = (lc_df['jdobs'].iloc[obs].values - t0)/(1+z)
    flux = lc_df['Fpsf'].iloc[obs].values
    flux_unc = lc_df['eFpsf'].iloc[obs].values
    filt_arr = lc_df['filter'].iloc[obs].values

    guess_0 = [-t_fl, 
               0, 2*np.max(flux[filt_arr == 'g']), 18, 2, 2, 2,
               0, 2*np.max(flux[filt_arr == 'r']), 18, 2, 2, 2
              ]
    
    pre_sec_peak = np.where(time <= 7)
    f_data = flux[pre_sec_peak]
    t_data = time[pre_sec_peak]
    f_unc_data = flux_unc[pre_sec_peak]
    filt_data = filt_arr[pre_sec_peak]
    
    ml_res = minimize(multifilter_nll, guess_0, method='Powell', # Powell method does not need derivatives
                      args=(f_data, t_data, f_unc_data, filt_data))
    ml_guess = ml_res.x

    
    #number of walkers
    nwalkers = nwalkers
    nfac = np.ones_like(ml_guess)*5e-3
    ndim = len(ml_guess) 

    # file to save samples
    filename = mcmc_h5_file
    backend = emcee.backends.HDFBackend(filename)
    backend.reset(nwalkers, ndim)        

    #initial position of walkers
    pos = [ml_guess + nfac * np.random.randn(ndim) for i in range(nwalkers)]

    with Pool() as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, 
                                        multifilter_lnposterior_no_sig0, 
                                        args=(f_data, t_data, 
                                              f_unc_data, filt_data),
                                        backend=backend,
                                        pool=pool)
        max_samples = max_samples

        index = 0
        autocorr = np.empty(max_samples)
        old_tau = np.inf
        check_tau = 50000
        for sample in sampler.sample(pos, iterations=max_samples):
            if sampler.iteration % check_tau:
                continue
            tau = sampler.get_autocorr_time(tol=0)
            autocorr[index] = np.mean(tau)
            index += 1

            # Check convergence
            converged = np.all(tau * 100 < sampler.iteration)
            converged &= np.all(np.abs(old_tau - tau) / tau < 0.01)
            if converged:
                break
            old_tau = tau
    
    if old_tau != np.inf:
        print("Model ran {} steps with a final tau: {}".format(sampler.iteration, tau))

def fit_single_filter_lc(lc_df, t0=0, z=0, t_fl=18, 
           mcmc_h5_file="ZTF_SN.h5",
           max_samples=int(2e6),
           nwalkers=2000):
    '''Perform an MCMC fit to the light curve'''
    
    obs = np.where((lc_df['programid'] == 2.0) & 
                   (lc_df['nbaseline'] > 30))
    
    
    time = (lc_df['jdobs'].iloc[obs].values - t0)/(1+z)
    flux = lc_df['Fpsf'].iloc[obs].values
    flux_unc = lc_df['eFpsf'].iloc[obs].values
    filt_arr = lc_df['filter'].iloc[obs].values

    guess_0 = [-t_fl, 
               0, 2*np.max(flux[filt_arr == 'g']), 18, 2, 2, 2,
              ]
    
    pre_sec_peak = np.where(time <= 7)
    f_data = flux[pre_sec_peak][np.where(filt_arr == 'g')]
    t_data = time[pre_sec_peak][np.where(filt_arr == 'g')]
    f_unc_data = flux_unc[pre_sec_peak][np.where(filt_arr == 'g')]
    
    ml_res = minimize(nll, guess_0, method='Powell', # Powell method does not need derivatives
                      args=(f_data, t_data, f_unc_data))
    ml_guess = ml_res.x

    
    #number of walkers
    nwalkers = nwalkers
    nfac = np.ones_like(ml_guess)*5e-3
    ndim = len(ml_guess) 

    # file to save samples
    filename = mcmc_h5_file
    backend = emcee.backends.HDFBackend(filename)
    backend.reset(nwalkers, ndim)        

    #initial position of walkers
    pos = [ml_guess + nfac * np.random.randn(ndim) for i in range(nwalkers)]

    with Pool() as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, 
                                        lnposterior_no_sig0, 
                                        args=(f_data, t_data, 
                                              f_unc_data),
                                        backend=backend,
                                        pool=pool)
        max_samples = max_samples

        index = 0
        autocorr = np.empty(max_samples)
        old_tau = np.inf
        check_tau = 50000
        for sample in sampler.sample(pos, iterations=max_samples):
            if sampler.iteration % check_tau:
                continue
            tau = sampler.get_autocorr_time(tol=0)
            autocorr[index] = np.mean(tau)
            index += 1

            # Check convergence
            converged = np.all(tau * 100 < sampler.iteration)
            converged &= np.all(np.abs(old_tau - tau) / tau < 0.01)
            if converged:
                break
            old_tau = tau
    
    if old_tau != np.inf:
        print("Model ran {} steps with a final tau: {}".format(sampler.iteration, tau))
    
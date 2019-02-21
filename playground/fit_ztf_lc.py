import numpy as np
import pandas as pd
import emcee
import corner
from scipy.optimize import minimize

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

def lnposterior(theta, f, t, f_err):
    lnp = lnprior(theta)
    lnl = lnlikelihood(theta, f, t, f_err)
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
    
    if len(theta) % 7 != 1:
        raise RuntimeError('The correct number of parameters were not included')
    
    ln_l = 0
    for filt_num, filt in enumerate(np.unique(filt_arr)):
        theta_filt = np.append(theta[0], theta[1+7*filt_num:7+7*filt_num])
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

def multifilter_lnposterior(theta, f, t, f_err, filt_arr):
    lnp = multifilter_prior(theta, filt_arr)
    lnl = multifilter_lnlikelihood(theta, f, t, f_err, filt_arr)
    if not np.isfinite(lnl):
        return -np.inf
    if not np.isfinite(lnp):
        return -np.inf
    return lnl + lnp


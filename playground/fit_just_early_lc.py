import numpy as np
import pandas as pd
import emcee
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
    model = -np.inf*np.ones_like(f)
    model[pre_exp] = a
    
    time_term = (t[~pre_exp] - t_0)
    model[~pre_exp] = a + a_prime * (time_term)**alpha_r
    assert np.all(model > -np.inf),'fewer model values than flux values'
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

def multifcqfid_lnlike_simple(theta, f, t, f_err, fcqfid_arr):
    
    n_fcqid = len(np.unique(fcqfid_arr))
    n_filt = len(np.unique(np.unique(fcqfid_arr) % 10))
    
    if len(theta) != 1 + 2*n_filt + n_fcqid:
        raise RuntimeError('The correct number of parameters were not included')
    
    ln_l = 0
    for fcqfid_num, fcqfid in enumerate(np.unique(fcqfid_arr)):
        filt = int(fcqfid % 10)
                
        theta_fcqfid = np.array([theta[0], theta[1 + 2*n_filt + fcqfid_num], 
                                 theta[2*filt-1], theta[2*filt]])
        fcqfid_obs = np.where(fcqfid_arr == fcqfid)
        f_fcqfid = f[fcqfid_obs]
        t_fcqfid = t[fcqfid_obs]
        f_err_fcqfid = f_err[fcqfid_obs]
        ln_l += lnlike_simple(theta_fcqfid, f_fcqfid, t_fcqfid, f_err_fcqfid)
    
    return ln_l

def multifcqfid_nll_simple(theta, f, t, f_err, fcqfid_arr):
    return -1*multifcqfid_lnlike_simple(theta, f, t, f_err, fcqfid_arr)

def multifcqfid_lnprior_simple(theta, fcqfid_arr):
    
    n_fcqid = len(np.unique(fcqfid_arr))
    n_filt = len(np.unique(np.unique(fcqfid_arr) % 10))
    
    if len(theta) != 1 + 2*n_filt + n_fcqid:
        raise RuntimeError('The correct number of parameters were not included')
    
    ln_p = 0
    for fcqfid_num, fcqfid in enumerate(np.unique(fcqfid_arr)):
        filt = int(fcqfid % 10)
                
        theta_fcqfid = np.array([theta[0], theta[1 + 2*n_filt + fcqfid_num], 
                                 theta[2*filt-1], theta[2*filt]])
        ln_p += lnprior_simple(theta_fcqfid)
    return ln_p

def multifcqfid_lnposterior_simple(theta, f, t, f_err, fcqfid_arr):
    lnp = multifcqfid_lnprior_simple(theta, fcqfid_arr)
    lnl = multifcqfid_lnlike_simple(theta, f, t, f_err, fcqfid_arr)
    if not np.isfinite(lnl):
        return -np.inf
    if not np.isfinite(lnp):
        return -np.inf
    return lnl + lnp

# multiplier term for the uncertainties
def lnlike_big_unc(theta, f, t, f_err, prior='uninformative'):
    if (prior == 'uninformative' or 
        prior == 'delta_alpha' or 
        prior == 'alpha_r_plus_colors'):
        t_0, a, a_prime, alpha_r, f_sigma = theta
    elif prior == 'delta2':
        alpha_r = 2
        t_0, a, a_prime, f_sigma = theta
        
    pre_exp = np.logical_not(t > t_0)
    model = -np.inf*np.ones_like(f)
    model[pre_exp] = a
    
    time_term = (t[~pre_exp] - t_0)
    model[~pre_exp] = a + a_prime * (time_term)**alpha_r * 10.**(-alpha_r)
    assert np.all(model > -np.inf),"fewer model values than flux values\n{}\n{}\na{}A'{}alpha{}f_sigma{}".format(model, time_term,a,a_prime,alpha_r,f_sigma)
    
    ln_l = -0.5*np.sum((f - model)**2 / ((f_sigma*f_err)**2)) - np.sum(np.log(f_sigma*f_err)) - 0.5*len(model)*np.log(np.sqrt(2*np.pi))
    return ln_l

def nll_big_unc(theta, flux, time, flux_err, prior='uninformative'):
    return -1*lnlike_big_unc(theta, flux, time, flux_err, prior=prior)

#Define priors on parameters  
def lnprior_big_unc(theta, prior='uninformative'):
    if (prior == 'uninformative' or 
        prior == 'delta_alpha' or 
        prior == 'alpha_r_plus_colors'):
        t_0, a, a_prime, alpha_r, f_sigma = theta
    elif prior == 'delta2':
        alpha_r = 2
        t_0, a, a_prime, f_sigma = theta
    if (a_prime < 0 or 
        f_sigma < 0 or 
        t_0 < -100 or
        t_0 > 0 or 
        alpha_r < 0 or
        alpha_r > 1e8 or
        a < -1e8 or
        a > 1e8):
        return -np.inf
    elif (prior == 'uninformative' or 
          prior == 'delta_alpha' or 
          prior == 'alpha_r_plus_colors'):
        return -np.log(a_prime) - np.log(f_sigma) - alpha_r*np.log(10)
    elif prior == 'delta2':
        return -np.log(a_prime) - np.log(f_sigma)

def lnposterior_big_unc(theta, flux, time, flux_err, prior='uninformative'):
    lnp = lnprior_big_unc(theta, prior=prior)
    if not np.isfinite(lnp):
        return -np.inf
    lnl = lnlike_big_unc(theta, flux, time, flux_err, prior=prior)
    if not np.isfinite(lnl):
        return -np.inf
    return lnl + lnp

def multifcqfid_lnlike_big_unc(theta, f, t, f_err, fcqfid_arr,
                               prior='uninformative'):
    
    n_fcqid = len(np.unique(fcqfid_arr))
    n_filt = len(np.unique(np.unique(fcqfid_arr) % 10))

    if ((prior == 'uninformative' or 
        prior == 'delta_alpha' or 
        prior == 'alpha_r_plus_colors') and  
        len(theta) != 1 + 2*n_filt + 2*n_fcqid):
        raise RuntimeError('Incorrect number of parameters entered')
    elif prior == 'delta2' and  len(theta) != 1 + n_filt + 2*n_fcqid:
        raise RuntimeError('Incorrect number of parameters entered')

    ln_l = 0
    for fcqfid_num, fcqfid in enumerate(np.unique(fcqfid_arr)):
        filt = int(fcqfid % 10)

        if (prior == 'uninformative' or 
            prior == 'delta_alpha' or 
            prior == 'alpha_r_plus_colors'):
            theta_fcqfid = np.array([theta[0], 
                                     theta[1 + 2*n_filt + 2*fcqfid_num], 
                                     theta[2*filt-1], 
                                     theta[2*filt],
                                     theta[2 + 2*n_filt + 2*fcqfid_num]])
        elif prior == 'delta2':
            theta_fcqfid = np.array([theta[0], 
                                     theta[1 + n_filt + 2*fcqfid_num], 
                                     theta[filt],
                                     theta[2 + n_filt + 2*fcqfid_num]])
            
        fcqfid_obs = np.where(fcqfid_arr == fcqfid)
        f_fcqfid = f[fcqfid_obs]
        t_fcqfid = t[fcqfid_obs]
        f_err_fcqfid = f_err[fcqfid_obs]
        ln_l += lnlike_big_unc(theta_fcqfid, f_fcqfid, t_fcqfid, f_err_fcqfid,
                               prior=prior)
    
    return ln_l

def multifcqfid_nll_big_unc(theta, f, t, f_err, fcqfid_arr,
                            prior='uninformative'):
    return -1*multifcqfid_lnlike_big_unc(theta, f, t, f_err, fcqfid_arr, 
                                         prior=prior)

def multifcqfid_lnprior_big_unc(theta, fcqfid_arr,
                                prior='uninformative'):
    
    n_fcqid = len(np.unique(fcqfid_arr))
    n_filt = len(np.unique(np.unique(fcqfid_arr) % 10))

    if ((prior == 'uninformative' or 
         prior == 'delta_alpha' or 
         prior == 'alpha_r_plus_colors') and  
        len(theta) != 1 + 2*n_filt + 2*n_fcqid):
        raise RuntimeError('Incorrect number of parameters entered')
    elif prior == 'delta2' and  len(theta) != 1 + n_filt + 2*n_fcqid:
        raise RuntimeError('Incorrect number of parameters entered')

    ln_p = 0
    for fcqfid_num, fcqfid in enumerate(np.unique(fcqfid_arr)):
        filt = int(fcqfid % 10)

        if (prior == 'uninformative' or 
            prior == 'delta_alpha' or 
            prior == 'alpha_r_plus_colors'):
            theta_fcqfid = np.array([theta[0], 
                                     theta[1 + 2*n_filt + 2*fcqfid_num], 
                                     theta[2*filt-1], 
                                     theta[2*filt],
                                     theta[2 + 2*n_filt + 2*fcqfid_num]])
        elif prior == 'delta2':
            theta_fcqfid = np.array([theta[0], 
                                     theta[1 + n_filt + 2*fcqfid_num], 
                                     theta[filt],
                                     theta[2 + n_filt + 2*fcqfid_num]])
        ln_p += lnprior_big_unc(theta_fcqfid, 
                                prior=prior)
    if prior == 'delta_alpha' or prior == 'alpha_r_plus_colors':
        ln_p += np.log((2*np.pi*0.09**2)**(-0.5) * 
                       np.exp((-0.18 - (theta[4]-theta[2]))**2/(-2*0.09**2)))
    if prior == 'alpha_r_plus_colors':
        ln_p += np.log((2*np.pi*0.1**2)**(-0.5) * 
                        np.exp((2 - theta[4])**2/(-2*0.1**2)))
    
    return ln_p

def multifcqfid_lnposterior_big_unc(theta, f, t, f_err, fcqfid_arr,
                                    prior='uninformative'):
    lnp = multifcqfid_lnprior_big_unc(theta, fcqfid_arr, 
                                      prior=prior)
    if not np.isfinite(lnp):
        return -np.inf
    lnl = multifcqfid_lnlike_big_unc(theta, f, t, f_err, fcqfid_arr, 
                                     prior=prior)
    if not np.isfinite(lnl):
        return -np.inf
    return lnl + lnp

def fit_lc(t_data, f_data, f_unc_data, fcqfid_data, 
           t_fl=18,
           mcmc_h5_file="ZTF_SN.h5",
           max_samples=int(2e6),
           nwalkers=100,
           rel_flux_cutoff = 0.5,
           ncores=None,
           emcee_burnin=True,
           use_emcee_backend=True,
           thin_by=1, 
           prior='uninformative'):
    '''Perform an MCMC fit to the light curve'''
    t_mcmc_start = time.time()
    
    if ncores == None:
        ncores = cpu_count() - 1
    
    n_filt = len(np.unique(np.unique(fcqfid_data) % 10))
    if (prior == 'uninformative' or 
        prior == 'delta_alpha' or 
        prior == 'alpha_r_plus_colors'):
        guess_0 = np.append([-t_fl] + [6e1, 2]*n_filt,
                            [1,1]*len(np.unique(fcqfid_data)))
    elif prior == 'delta2':
        guess_0 = np.append([-t_fl] + [6e1]*n_filt,
                            [1,1]*len(np.unique(fcqfid_data)))
        
    
    ml_res = minimize(multifcqfid_nll_big_unc, guess_0, method='Powell', # Powell method does not need derivatives
                      args=(f_data, t_data, f_unc_data, fcqfid_data, prior))
    ml_guess = ml_res.x
    
    ndim = len(ml_guess)
    nfac = [10**(-2.5)]*ndim

    #initial position of walkers
    rand_pos = [1 + nfac*np.random.randn(ndim) for i in range(nwalkers)]
    if ml_guess[0] < -5 and ml_guess[-1] < 5:
        pos = ml_guess*rand_pos
    else:
        pos = guess_0*rand_pos

    with Pool(ncores) as pool:
        if emcee_burnin:
            burn_sampler = emcee.EnsembleSampler(nwalkers, ndim, 
                                                multifcqfid_lnposterior_big_unc, 
                                                args=(f_data, t_data, 
                                                      f_unc_data, fcqfid_data, prior),
                                                pool=pool)
            burn_sampler.run_mcmc(pos, nsteps=50, 
                                thin_by=thin_by, progress=False)
            flat_burn_chain = burn_sampler.get_chain(flat=True)
            flat_burn_prob = np.argmax(burn_sampler.get_log_prob(flat=True))
            max_prob = flat_burn_chain[flat_burn_prob]
            pos = max_prob*rand_pos

        if use_emcee_backend:
            # file to save samples
            filename = mcmc_h5_file
            backend = emcee.backends.HDFBackend(filename)
            backend.reset(nwalkers, ndim)        

            sampler = emcee.EnsembleSampler(nwalkers, ndim, 
                                            multifcqfid_lnposterior_big_unc, 
                                            args=(f_data, t_data, 
                                                  f_unc_data, fcqfid_data, prior),
                                            pool=pool, backend=backend)
        else:
            sampler = emcee.EnsembleSampler(nwalkers, ndim, 
                                            multifcqfid_lnposterior_big_unc, 
                                            args=(f_data, t_data, 
                                                  f_unc_data, fcqfid_data, prior),
                                            pool=pool)
            
        max_samples = max_samples

        old_tau = np.inf
        for sample in sampler.sample(pos, 
                                     iterations=max_samples, 
                                     thin_by=thin_by, progress=False):
            if sampler.iteration <= int(1e3/thin_by):
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

def continue_chains(t_data, f_data, f_unc_data, fcqfid_data,
                    mcmc_h5_file="ZTF_SN.h5",
                    max_samples=int(2e6),
                    rel_flux_cutoff = 0.5,
                    ncores=None,
                    thin_by=1,
                    prior='uninformative'):
    '''Run MCMC for longer than initial fit'''
    t_mcmc_start = time.time()
    
    if ncores == None:
        ncores = cpu_count() - 1
    
    with Pool(ncores) as pool:
        # file to save samples
        filename = mcmc_h5_file
        new_backend = emcee.backends.HDFBackend(filename)
        _, nwalkers, ndim = np.shape(new_backend.get_chain())
        new_sampler = emcee.EnsembleSampler(nwalkers, ndim, 
                                        multifcqfid_lnposterior_big_unc,
                                        args=(f_data, t_data, 
                                              f_unc_data, fcqfid_data, prior),
                                        pool=pool, backend=new_backend)
        max_samples = max_samples
        steps_so_far = new_sampler.iteration
        old_tau = new_sampler.get_autocorr_time(tol=0)
        for i in range(int(max_samples/(2e4/thin_by))):
            new_sampler.run_mcmc(None, int(2e4/thin_by), 
                                 thin_by=thin_by, progress=False)
            tstart = time.time()
            tau = new_sampler.get_autocorr_time(tol=0)
            tend = time.time()
            steps_so_far = new_sampler.iteration
            print('''After {:d} steps, 
                     autocorrelation takes {:.3f} s ({} total FFTs)                
                     acceptance fraction = {:.4f}, and
                     tau = {}'''.format(steps_so_far, 
                       tend-tstart, nwalkers*ndim,
                       np.mean(new_sampler.acceptance_fraction), 
                       tau))
            # Check convergence
            converged = np.all(tau * 100 < new_sampler.iteration)
            converged &= np.all(np.abs(old_tau - tau) / tau < 0.01)
            if converged:
                break
            old_tau = tau


    print("Ran {} steps; final tau= {}".format(steps_so_far*thin_by, tau))
    t_mcmc_end = time.time()
    print("All in = {:.2f} s on {} cores".format(t_mcmc_end - t_mcmc_start, 
                                                 ncores))

def prep_light_curve(lc_hdf,
                     t_max=0, 
                     z=0,
                     g_max=1,
                     r_max=1,
                     rel_flux_cutoff=0.4, 
                     flux_scale = 100,
                     return_masked=False):
    
    # light curve data
    lc_df = pd.read_hdf(lc_hdf)
    time_rf = (lc_df['jdobs'].values - t_max)/(1+z)        
    baseline = np.where(time_rf < -20)
    has_baseline = np.ones_like(time_rf).astype(bool)
    fmcmc = lc_df['Fmcmc'].values
    fmcmc_unc = lc_df['Fmcmc_unc'].values
    zp = lc_df.zp.values
    zp_unc = lc_df.ezp.values
    
    f_zp = np.zeros_like(fmcmc)
    f_zp_unc = np.zeros_like(fmcmc)
    zp_base = np.zeros_like(fmcmc)
    
    for fcqfid in np.unique(lc_df.fcqfid.values):
        this_chip = np.where(lc_df.fcqfid.values == fcqfid)
        this_baseline = np.intersect1d(baseline, this_chip)
        
        if len(this_baseline) >= 1:
            zp_base[this_chip] = np.median(fmcmc[this_baseline]/10**(0.4*zp[this_baseline]))

            f_zp[this_chip] = fmcmc[this_chip]/10**(0.4*zp[this_chip])
            f_zp_unc[this_chip] = np.hypot(fmcmc_unc[this_chip]/10**(0.4*zp[this_chip]), 
                                             np.log(10)/2.5*fmcmc[this_chip]*zp_unc[this_chip]/10**(0.4*zp[this_chip]))
        else:
            has_baseline[this_chip] = 0
            
    g_obs = np.where((lc_df['filter'] == b'g') & (has_baseline))
    r_obs = np.where((lc_df['filter'] == b'r') & (has_baseline))

    f_zp[g_obs] = f_zp[g_obs]/g_max
    f_zp[r_obs] = f_zp[r_obs]/r_max
    f_zp_unc[g_obs] = f_zp_unc[g_obs]/g_max
    f_zp_unc[r_obs] = f_zp_unc[r_obs]/r_max
    
    new_night = np.append(np.where(np.diff(lc_df['jdobs'].values[has_baseline]) >= 0.6), 
                          len(lc_df['jdobs'].values[has_baseline])-1)

    mean_rf = np.zeros_like(new_night).astype(float)
    mean_g = np.zeros_like(new_night).astype(float)
    mean_r = np.zeros_like(new_night).astype(float)

    for nnumber, nidx in enumerate(new_night + 1):
        if nnumber == 0:
            start_idx = 0
        else:
            start_idx = new_night[nnumber-1] + 1
        end_idx = nidx


        jd_tonight = lc_df['jdobs'].values[has_baseline][start_idx:end_idx]
        fcqfid_tonight = lc_df.fcqfid.values[has_baseline][start_idx:end_idx]
        f_zp_tonight = f_zp[start_idx:end_idx]
        f_zp_unc_tonight = f_zp_unc[start_idx:end_idx]
        zp_base_tonight = zp_base[start_idx:end_idx]

        g_tonight = np.array(fcqfid_tonight % 2).astype(bool)

        mean_rf[nnumber] = np.mean((jd_tonight - t_max)/(1+z))
        if sum(g_tonight) > 0:
            mean_g[nnumber] = np.average(f_zp_tonight[g_tonight] - zp_base_tonight[g_tonight]/g_max, 
                                         weights=f_zp_unc_tonight[g_tonight]**(-2))
        if sum(g_tonight)/len(g_tonight) < 1:
            mean_r[nnumber] = np.average(f_zp_tonight[~g_tonight] - zp_base_tonight[~g_tonight]/r_max, 
                                         weights=f_zp_unc_tonight[~g_tonight]**(-2))

    cutoff_g = np.where((mean_rf < 0) & (mean_g > 0) & 
                       (mean_g < rel_flux_cutoff))
    t_cut_g = mean_rf[cutoff_g[0][-1]] + 0.5
    early_g = np.where(time_rf[g_obs] < t_cut_g)
    cutoff_r = np.where((mean_rf < 0) & (mean_r > 0) & 
                       (mean_r < rel_flux_cutoff))
    t_cut_r = mean_rf[cutoff_r[0][-1]] + 0.5
    early_r = np.where(time_rf[r_obs] < t_cut_r)
    early_obs = np.append(g_obs[0][early_g], r_obs[0][early_r])

    return_obs = np.intersect1d(np.where(has_baseline > 0), early_obs)
    not_included = np.setdiff1d(range(len(f_zp)), return_obs)    

    if not return_masked:
        return time_rf[return_obs], f_zp[return_obs]*flux_scale, f_zp_unc[return_obs]*flux_scale, lc_df.fcqfid.values[return_obs]
    else:
        return time_rf, f_zp*flux_scale, f_zp_unc*flux_scale, lc_df.fcqfid.values, return_obs


if __name__== "__main__":
    ztf_name = str(sys.argv[1])
    ncores = 27
    nsteps = int(1e6)
    thin_by = int(1)
    data_path = "/projects/p30796/ZTF/early_Ia/forced_lightcurves/sample_lc_v2/"
    backend_filename = data_path + "/{}_emcee.h5".format(ztf_name)
    use_emcee_backend = True
    rel_flux_cutoff=0.5
    prior='uninformative'
    
    if len(sys.argv) > 2:
        ncores = int(sys.argv[2])
    if len(sys.argv) > 3:
        nsteps = int(sys.argv[3])
    if len(sys.argv) > 4:
        thin_by = int(sys.argv[4])
    if len(sys.argv) > 5:
        backend_filename = str(sys.argv[5])
    if len(sys.argv) > 6:
        rel_flux_cutoff = float(sys.argv[6]) 
    if len(sys.argv) > 7:
        prior = str(sys.argv[7])
    if len(sys.argv) > 8:
        use_emcee_backend = False  

    lc_hdf = data_path + "/{}_force_phot.h5".format(ztf_name)
    salt_df = pd.read_csv(data_path + "../../Nobs_cut_salt2_spec_subtype_pec.csv")

    t0 = float(salt_df['t0_g_adopted'][salt_df['name'] == ztf_name].values)
    z = float(salt_df['z_adopt'][salt_df['name'] == ztf_name].values)
    g_max = float(salt_df['fratio_gmax_2adam'][salt_df['name'] == ztf_name].values)
    r_max = float(salt_df['fratio_rmax_2adam'][salt_df['name'] == ztf_name].values)
    
    
    t_data, f_data, f_unc_data, fcqfid_data = prep_light_curve(lc_hdf,
                                                               t_max=t0, 
                                                               z=z,
                                                               g_max=g_max,
                                                               r_max=r_max,
                                                               rel_flux_cutoff=rel_flux_cutoff)
    
    fit_lc(t_data, f_data, f_unc_data, fcqfid_data,
           mcmc_h5_file=backend_filename, 
           max_samples=nsteps, 
           ncores=ncores,
           use_emcee_backend=use_emcee_backend,
           thin_by=thin_by,
           rel_flux_cutoff=rel_flux_cutoff,
           prior=prior
           )
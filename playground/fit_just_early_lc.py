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
    model = 999*np.ones_like(f)
    model[pre_exp] = a
    
    time_term = (t[~pre_exp] - t_0)
    model[~pre_exp] = a + a_prime * (time_term)**alpha_r
    assert np.all(model != 999.),'fewer model values than flux values'
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
    
def fit_lc(lc_df, t0=0, z=0, t_fl=18, 
           mcmc_h5_file="ZTF_SN.h5",
           max_samples=int(2e6),
           nwalkers=100,
           rel_flux_cutoff = 0.5,
           ncores=None,
           emcee_burnin=True,
           use_emcee_backend=True,
           thin_by=1,
           g_max=1,
           r_max=1):
    '''Perform an MCMC fit to the light curve'''
    t_mcmc_start = time.time()
    
    if ncores == None:
        ncores = cpu_count() - 1
    
    g_obs = np.where(lc_df['filter'] == b'g')
    r_obs = np.where(lc_df['filter'] == b'r')

    time_rf = (lc_df['jdobs'].values - t0)/(1+z)
    flux = lc_df['Fratio'].values
    if g_max == 1:
        g_max = np.max(lc_df['Fratio'].iloc[g_obs].values)
    if r_max == 1:
        r_max = np.max(lc_df['Fratio'].iloc[r_obs].values)
    flux[g_obs] = flux[g_obs]/g_max
    flux[r_obs] = flux[r_obs]/r_max
    flux_unc = lc_df['Fratio_unc'].values
    flux_unc[g_obs] = flux_unc[g_obs]/g_max
    flux_unc[r_obs] = flux_unc[r_obs]/r_max
    fcqfid_arr = lc_df['fcqfid'].values

    cutoff_g = np.where((time_rf[g_obs] < 0) & 
                       (flux[g_obs] < rel_flux_cutoff))
    t_cut_g = time_rf[g_obs][cutoff_g[0][-1]] + 0.5
    early_g = np.where(time_rf[g_obs] < t_cut_g)
    cutoff_r = np.where((time_rf[r_obs] < 0) & 
                       (flux[r_obs] < rel_flux_cutoff))
    t_cut_r = time_rf[r_obs][cutoff_r[0][-1]] + 0.5
    early_r = np.where(time_rf[r_obs] < t_cut_r)
    early_obs = np.append(g_obs[0][early_g], r_obs[0][early_r])

    f_data = flux[early_obs]
    t_data = time_rf[early_obs]
    f_unc_data = flux_unc[early_obs]
    fcqfid_data = fcqfid_arr[early_obs]
    
    n_filt = len(np.unique(np.unique(fcqfid_arr) % 10))
    guess_0 = np.append([-t_fl] + [6e-3, 2]*n_filt,
                        np.zeros(len(np.unique(fcqfid_data))))
    
    ml_res = minimize(multifcqfid_nll_simple, guess_0, method='Powell', # Powell method does not need derivatives
                      args=(f_data, t_data, f_unc_data, fcqfid_data))
    ml_guess = ml_res.x
    
    ndim = len(ml_guess)
    nfac = [10**(-2.5)]*ndim

    #initial position of walkers
    rand_pos = [1 + nfac*np.random.randn(ndim) for i in range(nwalkers)]
    pos = ml_guess*rand_pos

    with Pool(ncores) as pool:
        if emcee_burnin:
            burn_sampler = emcee.EnsembleSampler(nwalkers, ndim, 
                                                multifcqfid_lnposterior_simple, 
                                                args=(f_data, t_data, 
                                                      f_unc_data, fcqfid_data),
                                                pool=pool)
            burn_sampler.sample(pos, max_iterations=50, 
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
                                            multifcqfid_lnposterior_simple, 
                                            args=(f_data, t_data, 
                                                  f_unc_data, fcqfid_data),
                                            pool=pool, backend=backend)
        else:
            sampler = emcee.EnsembleSampler(nwalkers, ndim, 
                                            multifcqfid_lnposterior_simple, 
                                            args=(f_data, t_data, 
                                                  f_unc_data, fcqfid_data),
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

def continue_chains(lc_df, t0=0, z=0,
                    mcmc_h5_file="ZTF_SN.h5",
                    max_samples=int(2e6),
                    rel_flux_cutoff = 0.5,
                    ncores=None,
                    thin_by=1,
                    g_max=1,
                    r_max=1):
    '''Run MCMC for longer than initial fit'''
    t_mcmc_start = time.time()
    
    if ncores == None:
        ncores = cpu_count() - 1
    
    g_obs = np.where(lc_df['filter'] == b'g')
    r_obs = np.where(lc_df['filter'] == b'r')

    time_rf = (lc_df['jdobs'].values - t0)/(1+z)
    flux = lc_df['Fratio'].values
    if g_max == 1:
        g_max = np.max(lc_df['Fratio'].iloc[g_obs].values)
    if r_max == 1:
        r_max = np.max(lc_df['Fratio'].iloc[r_obs].values)
    flux[g_obs] = flux[g_obs]/g_max
    flux[r_obs] = flux[r_obs]/r_max
    flux_unc = lc_df['Fratio_unc'].values
    flux_unc[g_obs] = flux_unc[g_obs]/g_max
    flux_unc[r_obs] = flux_unc[r_obs]/r_max
    fcqfid_arr = lc_df['fcqfid'].values

    cutoff_g = np.where((time_rf[g_obs] < 0) & 
                       (flux[g_obs] < rel_flux_cutoff))
    t_cut_g = time_rf[g_obs][cutoff_g[0][-1]] + 0.5
    early_g = np.where(time_rf[g_obs] < t_cut_g)
    cutoff_r = np.where((time_rf[r_obs] < 0) & 
                       (flux[r_obs] < rel_flux_cutoff))
    t_cut_r = time_rf[r_obs][cutoff_r[0][-1]] + 0.5
    early_r = np.where(time_rf[r_obs] < t_cut_r)
    early_obs = np.append(g_obs[0][early_g], r_obs[0][early_r])

    f_data = flux[early_obs]
    t_data = time_rf[early_obs]
    f_unc_data = flux_unc[early_obs]
    fcqfid_data = fcqfid_arr[early_obs]

    with Pool(ncores) as pool:
        # file to save samples
        filename = mcmc_h5_file
        new_backend = emcee.backends.HDFBackend(filename)
        _, nwalkers, ndim = np.shape(new_backend.get_chain())
        new_sampler = emcee.EnsembleSampler(nwalkers, ndim, 
                                        multifcqfid_lnposterior_simple,
                                        args=(f_data, t_data, 
                                              f_unc_data, fcqfid_data),
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
import numpy as np
import pandas as pd
import emcee
from multiprocessing import Pool, cpu_count
import time
import sys
from fit_just_early_lc import fit_lc, prep_light_curve
from astropy.cosmology import FlatLambdaCDM
from astropy.coordinates import Distance


if __name__== "__main__":
    ztf_name = str(sys.argv[1])
    ncores = 27
    nsteps = int(1e6)
    thin_by = int(1)
    data_path = "/projects/p30796/ZTF/early_Ia/forced_lightcurves/sample_lc_v2/"
    backend_filename = data_path + "/{}_emcee.h5".format(ztf_name)
    rel_flux_cutoff=0.5
    z_target = 0.1
    
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
        z_target = float(sys.argv[7])  

    lc_hdf = data_path + "/{}_force_phot.h5".format(ztf_name)
    salt_df = pd.read_csv(data_path + "../../Nobs_cut_salt2_spec_subtype.csv")

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

    flat_h0_73 = FlatLambdaCDM(H0=73.24,Om0=0.275)
    d_l = Distance(z=z, cosmology=flat_h0_73)
    d_l_target = Distance(z=z_target, cosmology=flat_h0_73)
    snr_factor = (d_l.value/d_l_target.value)**0.5
    f_data *= snr_factor
    
    fit_lc(t_data, f_data, f_unc_data, fcqfid_data,
           mcmc_h5_file=backend_filename, 
           max_samples=nsteps, 
           ncores=ncores,
           use_emcee_backend=use_emcee_backend,
           thin_by=thin_by,
           rel_flux_cutoff=rel_flux_cutoff
           )
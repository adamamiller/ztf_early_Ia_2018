import numpy as np
import pandas as pd
import emcee
from multiprocessing import Pool, cpu_count
import time
import sys
from fit_just_early_lc import continue_chains, prep_light_curve

if __name__== "__main__":
    ztf_name = str(sys.argv[1])
    ncores = 27
    nsteps = int(1e6)
    thin_by = int(1)
    data_path = "/projects/p30796/ZTF/early_Ia/forced_lightcurves/sample_lc_v2/"
    backend_filename = data_path + "/{}_emcee.h5".format(ztf_name)
    rel_flux_cutoff=0.5
    
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
               
    continue_chains(t_data, f_data, f_unc_data, fcqfid_data,
                    mcmc_h5_file=backend_filename, 
                    max_samples=nsteps, 
                    ncores=ncores,
                    thin_by=thin_by,
                    rel_flux_cutoff=rel_flux_cutoff
                    )
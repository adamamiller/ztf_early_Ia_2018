import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

import emcee
import corner

def f_t(times, amplitude=25, t_0=0, alpha_r=2):
    
    return amplitude * (times - t_0)**alpha_r

def plot_both_filt(theta, 
                   t, f, f_unc, fcqfid_arr,
                   obs_for_model,
                   samples, samp_nums
                  ):

    color_dict = {b'r': 'Crimson',
                  b'g': 'MediumAquaMarine'}
    offset_dict = {b'r': -10,
                  b'g': 10}
    mark_color_dict = {b'r': 'white',
                  b'g': 'MediumAquaMarine'}
    sym_dict = {b'r': 'o',
                b'g': 'o'}
    mec_dict = {b'r': 'Crimson',
                b'g': '0.5'}
    mew_dict = {b'r': 2,
                b'g': 0.5}
    filt_dict = {1: b'g', 2: b'r'}
    
    plot_min = 0.0
    res_max = 0.0
    res_min = 0.0

    fig = plt.figure()
    axPlot = plt.axes([0.15, 0.37, 0.82, 0.60])
    axRes = plt.axes([0.15, 0.11, 0.82, 0.25], sharex=axPlot)
    
    n_fcqid = len(np.unique(fcqfid_arr[obs_for_model]))
    n_filt = len(np.unique(np.unique(fcqfid_arr[obs_for_model]) % 10))

    if len(theta) != 1 + 2*n_filt + 2*n_fcqid:
        raise RuntimeError('The correct number of parameters were not included')

    for fcqfid_num, fcqfid in enumerate(np.unique(fcqfid_arr)):
        filt_int = int(fcqfid % 10)
        filt = filt_dict[filt_int]
        theta_fcqfid = np.array([theta[0], theta[1 + 2*n_filt + 2*fcqfid_num], 
                                 theta[2*filt_int-1], theta[2*filt_int],
                                 theta[2 + 2*n_filt + 2*fcqfid_num]])

        fcqfid_obs = np.where(fcqfid_arr == fcqfid)
        f_fcqfid = f[fcqfid_obs]
        t_fcqfid = t[fcqfid_obs]
        f_err_fcqfid = f_unc[fcqfid_obs]        
        
        
        t_post = np.linspace(theta[0], 80, 1000)
        t_pre = np.linspace(min(t_fcqfid), theta[0], 1000)
        model_flux = f_t(t_post, theta_fcqfid[2], theta_fcqfid[0], theta_fcqfid[3]) # remove baseline from model
        
        fit_data = np.intersect1d(obs_for_model, fcqfid_obs)
        no_fit = np.setdiff1d(fcqfid_obs, fit_data)
                
        axPlot.errorbar(t[fit_data], f[fit_data] - theta_fcqfid[1] + offset_dict[filt], f_unc[fit_data],
                    fmt = sym_dict[filt], color=mark_color_dict[filt], ecolor=color_dict[filt],
                    mec=mec_dict[filt], mew=mew_dict[filt])
        axPlot.errorbar(t[no_fit], f[no_fit] - theta_fcqfid[1] + offset_dict[filt], f_unc[no_fit],
                    fmt = sym_dict[filt], color=mark_color_dict[filt], ecolor=color_dict[filt],
                    mec=mec_dict[filt], mew=mew_dict[filt], alpha=0.2)
        
        axPlot.plot(t_post, model_flux + offset_dict[filt], color=color_dict[filt], zorder=10)
        axPlot.plot(t_pre, np.zeros_like(t_pre) + offset_dict[filt], color=color_dict[filt], zorder=10)
        axPlot.set_xlim(-30, 1)
        
        plot_range = np.where((t > -30) & (t < 1) )
        res_range = np.where((t > -30) & (t < np.max(t[fit_data]) ))
        plot_points = np.intersect1d(plot_range, fcqfid_obs)
        res_points = np.intersect1d(res_range, fcqfid_obs)
        
        if len(plot_points) > 0:
            plot_min = np.min([min(f[plot_points] - theta_fcqfid[1] + offset_dict[filt]), plot_min])
        axPlot.set_ylim(plot_min-5, 110)

        after_exp = t >= theta_fcqfid[0]

        residuals = np.append(f[~after_exp] - theta_fcqfid[1], 
                              f[after_exp] - (theta_fcqfid[1] + f_t(t[after_exp], theta_fcqfid[2], theta_fcqfid[0], theta_fcqfid[3]))  
                             )
        # plot residuals
#         axRes.errorbar(t_fcqfid[half_max], residuals[half_max] + offset_dict[filt], f_err_fcqfid[half_max],
#                        fmt = sym_dict[filt], color=mark_color_dict[filt], ecolor=color_dict[filt],
#                        mec=mec_dict[filt], mew=mew_dict[filt])
#         axRes.errorbar(t_fcqfid[~half_max], residuals[~half_max] + offset_dict[filt], f_err_fcqfid[~half_max],
#                        fmt = sym_dict[filt], color=mark_color_dict[filt], ecolor=color_dict[filt],
#                        mec=mec_dict[filt], mew=mew_dict[filt], alpha=0.2)
#         axRes.plot([-5000,10000], [offset_dict[filt], offset_dict[filt]], '-', color=color_dict[filt])
#         axRes.set_ylim(min(residuals[half_max]) - 0.1, max(residuals[half_max]) + 0.1)

        #plot pull
        axRes.plot(t[fit_data], residuals[fit_data]/(f_unc[fit_data]*theta_fcqfid[4]),
                       sym_dict[filt], color=mark_color_dict[filt], 
                       mec=mec_dict[filt], mew=mew_dict[filt])
        axRes.plot(t[no_fit], residuals[no_fit]/(f_unc[no_fit]*theta_fcqfid[4]),
                       sym_dict[filt], color=mark_color_dict[filt], 
                       mec=mec_dict[filt], mew=mew_dict[filt], alpha=0.2)
        axRes.plot([-5000,10000], [0, 0], '-k')
        if len(res_points) > 0:
            res_min = np.min([min(residuals[res_points]/(f_unc[res_points]*theta_fcqfid[4])), res_min])
            res_max = np.max([max(residuals[res_points]/(f_unc[res_points]*theta_fcqfid[4])), res_max])
        axRes.set_ylim(res_min - 0.2, 
                       res_max + 0.2)

    axPlot.set_ylabel('$f + \mathrm{offset} \; (\mathrm{arbitrary \; units})$', fontsize=14)
    axPlot.xaxis.set_minor_locator(MultipleLocator(1))
    axPlot.yaxis.set_major_locator(MultipleLocator(20))
    axPlot.yaxis.set_minor_locator(MultipleLocator(10))
    axPlot.tick_params(right=True, top=True, bottom=False, which='both', labelsize=11)
    
    axRes.set_xlim(-30, 1)
    axRes.set_xlabel('$t - t_0 \; (\mathrm{restframe \; d})$', fontsize=14)
    axRes.set_ylabel('$\mathrm{pull}$', fontsize=14)
    axRes.xaxis.set_minor_locator(MultipleLocator(1))
    axRes.yaxis.set_minor_locator(MultipleLocator(np.mean(np.diff(axRes.get_yticks()))/2))
    axRes.yaxis.set_major_locator(MultipleLocator(np.mean(np.diff(axRes.get_yticks()))))
    axRes.tick_params(right=True,which='both', labelsize=11)

    # plot up t0
    t0_med = np.median(samples[:,0])
    t0_theta = theta_fcqfid[0]
    axPlot.plot([t0_med,t0_med], [-500,500], 
           '--', color='0.5', lw=1)
    axRes.plot([t0_med,t0_med], [-500,500], 
           '--', color='0.5', lw=1)
    axPlot.plot([t0_theta,t0_theta], [-500,500], 
           '--', color='0.8', lw=0.5)
    axRes.plot([t0_theta,t0_theta], [-500,500], 
           '--', color='0.8', lw=0.5)
    
    for samp_num in samp_nums:
        theta_samp = samples[samp_num]
        for fcqfid_num, fcqfid in enumerate(np.unique(fcqfid_arr)):
            filt_int = int(fcqfid % 10)
            filt = filt_dict[filt_int]
            theta_fcqfid = np.array([theta_samp[0], theta_samp[1 + 2*n_filt + 2*fcqfid_num], 
                                     theta_samp[2*filt_int-1], theta_samp[2*filt_int],
                                     theta_samp[2 + 2*n_filt + 2*fcqfid_num]])

        
            t_post = np.linspace(theta_fcqfid[0], 80, 1000)
            t_pre = np.linspace(-80, theta_fcqfid[0], 1000)
            model_flux =  f_t(t_post, theta_fcqfid[2], theta_fcqfid[0], theta_fcqfid[3])
            axPlot.plot(t_post, model_flux + offset_dict[filt], '--', 
                        color=color_dict[filt], zorder=10,
                        alpha=0.4)
            axPlot.plot(t_pre, np.zeros_like(t_pre) + offset_dict[filt], '--',
                        color=color_dict[filt], zorder=10,
                        alpha=0.4)
    plt.setp(axPlot.get_xticklabels(), visible=False)

    return fig
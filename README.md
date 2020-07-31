# ztf_early_Ia_2018

This (terribly organized) repository contains the software necessary to replicate the analysis presented in [ZTF Early Observations of Type Ia Supernovae II: First Light, the Initial Rise, and Time to Reach Maximum Brightness](https://ui.adsabs.harvard.edu/abs/2020arXiv200100598M/abstract) by Miller, Yao, Bulla, et al. It is largely a collection of a few python scrips (necessary to fit the early rise of the SNe Ia) and several jupyter notebooks that were used to create the plots in the paper. 

A rough map of the relevant files is as follows:

 * [fit_just_early_lc.py](playground/fit_just_early_lc.py) –– main script for fitting the power-law rise (*warning* there are many hard coded links that make it difficult to simply apply these models to other data sets)
 * [mean_alpha.ipynb](playground/mean_alpha.ipynb) –– PDFs of model parameters (Figs 4, 5, 6) 
 * [PlotPlayground.ipynb](playground/PlotPlayground.ipynb) –– correlation plots (Figs 7, 9, 10) 
 * [first_light_vs_first_detection.ipynb](playground/first_light_vs_first_detection.ipynb) –– systematic underestimation of rise time as a function of z (Fig 8) 
 * [analyze_tsquared.ipynb](playground/analyze_tsquared.ipynb) –– results for the alpha=2 prior (Figs 11, 12, 13) 
 * [MakeTables.ipynb](playground/MakeTables.ipynb) –– creates the TeX tables in the paper
 * [VolumeLimited.ipynb](playground/VolumeLimited.ipynb) –– calculates mean values of model parameters for a volume limited sample of SNe
 * [quality_assurance.ipynb](playground/quality_assurance.ipynb) –– assess the quality of individual model fits (Fig B2)
 * [CornerLC.ipynb](playground/CornerLC.ipynb) –– make final corner and light curve plots (Figs 1, 2, 3, B1, C1)
 * [Systematic_frac_flux.ipynb](playground/Systematic_frac_flux.ipynb) –– show systematic change in alpha and t_rise as a function of the fraction of flux being fit in the light curve 
# ztf_early_Ia_2018

This (terribly organized) repository contains the software necessary to replicate the analysis presented in [ZTF Early Observations of Type Ia Supernovae II: First Light, the Initial Rise, and Time to Reach Maximum Brightness](https://ui.adsabs.harvard.edu/abs/2020arXiv200100598M/abstract) by Miller, Yao, Bulla, et al. It is largely a collection of a few python scrips (necessary to fit the early rise of the SNe Ia) and several jupyter notebooks that were used to create the plots in the paper. 

A rough map of the relevant files is as follows:

 * [fit_just_early_lc.py](playground/fit_just_early_lc.py) –– main script for fitting the power-law rise
 * [mean_alpha.ipynb](playground/mean_alpha.ipynb) –– PDFs of model parameters (Figs 4, 5, 6) 
 * [PlotPlayground.ipynb](playground/PlotPlayground.ipynb) –– correlation plots (Figs 7, 9, 10) 
 * [playground/first_light_vs_first_detection.ipynb](playground/first_light_vs_first_detection.ipynb) –– systematic underestimation of rise time as a function of z (Fig 8) 
 * [analyze_tsquared.ipynb](playground/analyze_tsquared.ipynb) –– results for the alpha=2 prior (Figs 11, 12) 
 * [MakeTables.ipynb](playground/MakeTables.ipynb) –– creates the TeX tables in the paper
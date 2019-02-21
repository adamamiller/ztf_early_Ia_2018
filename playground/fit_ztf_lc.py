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
import fnmatch
import glob
import sys
import subprocess
import re

import numpy as np
import pandas as pd

from EASI_analysis import data as easi_data
from EASI_analysis import signal as easi_signal

def getBiweightLocation(samples, n_iters=100):
    """
    Computes the biweight location of the input samples.

    Parameters:
    samples (numpy.ndarray): 1D array of sample values.
    n_iters (int): Number of iterations for the biweight location computation.

    Returns:
    float: The biweight location of the input samples.
    """
    wts = np.ones(samples.size)
    c = 9 * np.median( np.abs( samples - np.median( samples ) ) )
    for _ in range( n_iters ):
        if wts.sum()==0: 
            wts = np.ones(samples.size) # reset misbehaving weights
        M = ( samples * wts ).sum() / wts.sum()
        chi = ( samples - M ) / c
        wts = ( 1 - chi**2 )**2
        wts = wts * ( np.abs( chi ) <= 1 ).astype( float )
    return M

def getBiweightLocationOfWaveforms( waveforms, n_iters=100 ):
    '''
    waveforms is an array of size MxN, representing 
    M waveforms of size N. Return value is a waveform of size N 
    which is the point-wise biweight locaiton of all waveforms. 
    '''
    wts = np.ones( waveforms.shape ) # MxN
    c = 9 * np.median( 
        np.abs( 
            waveforms - np.median( waveforms, axis=0, keepdims=True ).repeat( waveforms.shape[0], axis=0 ) 
        ), 
        axis=0 
    ) # N
    for _ in range( n_iters ):
        wts[ :, wts.sum( axis=0 )==0] = 1. # reset misbehaving weights
        M = ( waveforms * wts ).sum( axis=0 ) / wts.sum( axis=0 ) # N
        chi = ( waveforms - M[np.newaxis,:].repeat( waveforms.shape[0], axis=0 ) ) / c[np.newaxis,:].repeat( waveforms.shape[0], axis=0 ) # MxN
        wts = ( 1 - chi**2 )**2 # MxN
        wts = wts * ( np.abs( chi ) <= 1 ).astype( float ) # MxN
    return M

# LG2170.py
# This module contains routines specific to the analysis of LG 2710 cylindrical cells

import fnmatch
import glob
import sys
import subprocess

import numpy as np
import pandas as pd
import scipy.optimize as sciopt
import scipy.signal as spsig
from scipy.signal import argrelextrema, find_peaks
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tqdm.auto import tqdm
import tftb

import h5py as h5

from logzero import logger
from tqdm.auto import tqdm

import functools as ftls

# EASI imports
from EASI_analysis import data as easi_data
from EASI_analysis import signal as easi_signal

def GenerateFullRotationPlot( signal_database, serial_number, measurement_angle, movie_name=None ):
    sig_grouped = signal_database.groupby( [ 'serial_number', 'angle' ] )
    waves = sig_grouped.get_group( ( serial_number, measurement_angle ) ).sort_values( by='rotation' )
    waveforms = [ wav for wav in waves[ 'wave' ] ]
    waveforms = [ wv - wv.mean() for wv in waveforms ] 
    rotation_map = np.concatenate( [ wv[np.newaxis,:] for wv in waveforms ], axis=0 )
    return waveforms, rotation_map

def WignerToLogScale( wigner, Mn, Mx ):
    # wigner[ np.where( np.abs( wigner ) < 1.e-10 * np.abs( wigner ).min() ) ] = 0.
    wigner_sign = np.sign( wigner )
    wigner_log_transform = wigner_sign * np.log( sys.float_info.min + wigner_sign*wigner )
    Mn, Mx = [ np.sign( m ) * np.log( sys.float_info.min + np.sign( m )*m ) for m in [ Mn, Mx ] ]
    return wigner_log_transform, Mn, Mx

def load_theta(name, Nsamples, NumTransducers ):
    tmp = np.load( name ) # load npz file
    g = ( tmp['f_GainDB'] + 12* tmp['SuperGainOn'] ) # calculate the gain used
    data = tmp['data'][:Nsamples*NumTransducers].reshape( NumTransducers, Nsamples ).T # reshape data in [Nt x 10] format
    K = ( 2**-13 )* 10**( -(g-8)/20 )*1000 # calculate gain
    t = np.arange( Nsamples) / 125 * tmp['decimationFactor']+tmp['f_RxDelay']
    return data*K , t

def LoadData( 
    path_template,          # formattable string for datafile path 
    barcodes,               # list of cell barcodes/serial numbers
    num_rotation_steps,     # number of angular steps over 360 deg
    step_size,              # step size in degrees
    angles,                 # list containing the angular separation between transmitting and receiving transducer
    num_transducers,        # number of transducers in the array
    config_type,            # string that specifies a configuration
    sort_key,               # function handle that specifies how to sort the scan files in order ot rotation angle
    time_unit,              # unit of time, typically 1 us. 
    num_samples             # number of time-domain samples
):
    signals = []
    for bc in barcodes: 
        fname = path_template%bc
        files = fnmatch.filter( glob.glob( fname ), config_type )
        logger.info( f'Found {len( files )} files for cell {bc}. ' )
        data = np.zeros( ( num_samples, num_transducers, num_rotation_steps ) )
        dd = { st:[] for st in [ 'serial_number', 'angle', 'rotation', 'wave' ] }
        for n, file in enumerate( files ):
            rot = sort_key( file ) * step_size 
            tmp, t = load_theta( file, num_samples, num_transducers )
            tmp -= tmp.mean( axis=0, keepdims=True ).repeat( tmp.shape[0], axis=0 )
            data[:,:,n] = tmp
            for i, a in enumerate( angles ): 
                dd[ 'serial_number' ].append( str( bc ) )
                dd[ 'angle' ].append( a )
                dd[ 'rotation' ].append( rot )
                dd[ 'wave' ].append( tmp[:,i] )
        df = pd.DataFrame().from_dict( dd )
        signals.append( df )
    signals = pd.concat( signals )
    signals = easi_data.Signals( signals )
    signals.t = t * time_unit
    return signals

def GenerateAmplitudePolarPlot( sigs, cell_list, angle_list, roll_list, ax=None ):
    if 'amplitude_p2p' not in sigs.columns: 
        sigs[ 'amplitude_p2p' ] = [ 
            wav.max() - wav.min()
            for wav in list( sigs[ 'wave' ].to_numpy() )
        ]
    sigs_bycell = sigs.groupby( [ 'serial_number', 'angle' ] )
    if ax is None: 
        ax = plt.figure().subplots( subplot_kw={ 'projection':'polar' } )
    ax.clear()
    ax.set_theta_offset( -np.pi/2 )
    
    for cell, angle, roll in zip( cell_list, angle_list, roll_list ):
        grp = sigs_bycell.get_group( ( cell, angle ) )
        data = grp[ [ 'rotation', 'amplitude_p2p' ] ].sort_values( by='rotation' ).to_numpy()
        data[:,0] = ( data[:,0] - roll ) % 360.
        data = data.T
        ax.plot( data[0]*np.pi/180, data[1], '-', label='%s'%cell )
    ax.legend( fontsize=5 )
    return ax, data

def PairwiseAlign( t, wav1, wav2, window ):
    assert all( [ isinstance( w, float ) for w in window ] ) and len( window )==2, 'Window should contain time interval (a, b ). '
    here = np.where( np.logical_and( t > window[0], t < window[1] ) )[0]
    snp1, snp2 = [ w[ here ] for w in [ wav1, wav2 ] ]
    mx = max( [ snp.size for snp in [ snp1, snp2 ] ] )
    snp1, snp2 = [ np.pad( snp, ( mx, mx ) ) for snp in [ snp1, snp2 ] ]
    crosscorr = []
    for n in np.arange( -mx, mx+1 ):
        crosscorr.append( np.correlate( snp1, np.roll( snp2, n ) )[0] )
    myroll = np.argmax( crosscorr ) - mx
    return wav1, np.roll( wav2, myroll ), crosscorr

def ManualUnwrap( phase, discont=np.pi, discont_tol=0.01 ):
    dif = phase[1:] - phase[:-1]
    here = list( np.where( np.abs( np.abs( dif ) - discont ) < discont_tol*np.pi )[0] )
    here.append( len( phase )-1 )
    corrections = np.cumsum( dif[ here[:-1] ] )
    for n, ( strt, stop ) in enumerate( zip( here[:-1], here[1:] ) ):
        if strt+1 in here or strt-1 in here:
            continue # this is not an out-of-phase discontinuity
        phase[strt+1:stop+1] -= corrections[n]
    return phase

def GetAnalyticalParameters( wave ):
    wave_an = spsig.hilbert( wave )
    amp = np.abs( wave_an )
    phase = np.unwrap( np.angle( wave_an ) )
    phase = ManualUnwrap( phase )
    freq = ( 1/2/np.pi ) * ( phase[1:] - phase[:-1] )
    return amp, phase, freq

def getBiweightLocation(samples, n_iters=100, C=5. ):
    """
    Computes the biweight location of the input samples.

    Parameters:
    samples (numpy.ndarray): 1D array of sample values.
    n_iters (int): Number of iterations for the biweight location computation.

    Returns:
    float: The biweight location of the input samples.
    """
    c = np.finfo( float ).eps + C*np.median( np.abs( samples - np.median( samples ) ) )
    wts = np.ones(samples.size)
    for _ in range( n_iters ):
        if wts.sum()==0: 
            wts = np.ones(samples.size) # reset misbehaving weights
        M = ( samples * wts ).sum() / wts.sum()
        chi = ( samples - M ) / c
        wts = ( 1 - chi**2 )**2
        wts = wts * ( np.abs( chi ) <= 1 ).astype( float )
    return M

def getBiweightLocationOfWaveforms( waveforms, n_iters=100, C=5. ):
    '''
    waveforms is an array of size MxN, representing 
    M waveforms of size N. Return value is a waveform of size N 
    which is the point-wise biweight locaiton of all waveforms. 
    '''
    
    c = np.finfo( float ).eps + C*np.median( 
        np.abs( 
            waveforms - np.median( waveforms, axis=0, keepdims=True ).repeat( waveforms.shape[0], axis=0 ) 
        ), 
        axis=0 
    ) # N
    wts = np.ones( waveforms.shape ) # MxN
    for _ in range( n_iters ):
        wts[ :, wts.sum( axis=0 )==0] = 1. # reset misbehaving weights
        M = ( waveforms * wts ).sum( axis=0 ) / wts.sum( axis=0 ) # N
        chi = ( waveforms - M[np.newaxis,:].repeat( waveforms.shape[0], axis=0 ) ) / c[np.newaxis,:].repeat( waveforms.shape[0], axis=0 ) # MxN
        wts = ( 1 - chi**2 )**2 # MxN
        wts = wts * ( np.abs( chi ) <= 1 ).astype( float ) # MxN
    return M

def GetPeakWidthProfile( t, data, data_mean, watch_window=0.25e-6, peak_params={ 'height':3e7, 'width':1, 'distance':3 } ):
    '''
    use 'data_mean' to identify peak locations, then compute variations in 
    nominal peak locations as indicated by 'data' at each of these nominal locations. 
    This becomes the peak width profile. 
    '''
    peaks = find_peaks( data_mean, **peak_params ) # this picks all reasonable peaks in the desired region. 
    peak_locs = []
    for peak_loc in t[ peaks[0] ]:
        t1 = peak_loc - watch_window/2.
        t2 = peak_loc + watch_window/2. 
        my_slice_idx = list( np.where( np.logical_and( t > t1, t < t2 ) )[0] )
        this_peak_data = data[:,my_slice_idx] # if everything worked right, there should be one peak in each of these slices. 
        this_peak_locs = [] 
        for slc in this_peak_data:
            ind_peak = list( find_peaks( slc )[0] )
            # print( 'my_slice_idx = ', my_slice_idx )
            # print( 'ind_peak = ', ind_peak )
            try: 
                ind_peak = my_slice_idx[ ind_peak[0] ]
                this_peak_locs.append( t[ ind_peak ] ) # there should be only one
            except: 
                continue
        if this_peak_locs:
            peak_locs.append( this_peak_locs )
    return [ [ min( elem), max( elem ) ] for elem in peak_locs ], peak_locs

def mergeProfiles( p1, p2 ):
    '''
    Each input argument is a list of float pairs [ f1, f2 ] that denote an interval. 
    This function combined the lists into one list, with the profile_merged appearing 
    in order.  
    ''' 
    # prepare list to sort
    p = [ p1, p2 ]
    to_sort = [ [ val, 0, pos ] for pos, val in enumerate( [ n[0] for n in p1 ] ) ]
    to_sort.extend( [ [ val, 1, pos ] for pos, val in enumerate( [ n[0] for n in p2 ] ) ] )
    to_sort.sort( key=lambda x: x[0] ) # sort by left boundary of interval
    
    # create merged profile
    profile_merged = [ 
        p[ this[1] ][ this[2] ]
        for this in to_sort
    ] # profile_merged sorted by starting point
    
    def custom_merge( lst0, lst1 ):
        return [ lst0[0], lst1[1] ]

    for n in range( len( profile_merged ) ):
        try: 
            while profile_merged[n+1][0] < profile_merged[n][1]: # there is overlap
                if profile_merged[n+1][1] > profile_merged[n][1]:
                    profile_merged[n] = custom_merge( profile_merged[n], profile_merged[n+1] )
                del profile_merged[n+1]
        except: 
            continue
    return profile_merged

def load_theta( name, NumTransducers ):
    Nsamples = 8192
    tmp = np.load(name) # load npz file
    g = (tmp['f_GainDB']+12*tmp['SuperGainOn']) # calculate the gain used
    data = tmp['data'][0:Nsamples*NumTransducers].reshape((NumTransducers,Nsamples)).T # reshape data in [Nt x 10] format
    K = 2**-13*10**(-(g-8)/20)*1000 # calculate gain
    t = np.arange(Nsamples)/125*tmp['decimationFactor']+tmp['f_RxDelay']
    return data*K , t

def LoadScans( barcode, path, NumTransducers=3, N=100, angles=[ '48', '180', '132' ], step=5, sort_key=None ):
    signals = []
    for bc in barcode:
        bash_command_to_sort_files = f'find {path} -type f -printf \'%T@ %p\n\' | sort -n | cut -d\' \' -f2-', # sort by creation time, corresponding to rotations
        files = subprocess.run( 
            bash_command_to_sort_files, 
            shell=True, 
            capture_output=True, 
            text=True, 
            check=True
        ).stdout.strip().split( '\n' )

        files = [ f for f in files if '(1)' not in f ]
        files = files[:-1]
        if not sort_key: 
            files.sort( key=sort_key )
        
        print(f"Cell {bc} found {len(files)} files")
        
        data = np.zeros((8192,NumTransducers,N))# [time,Angle,rotation] -- Angle: angle from the TX, note that 135 is actually -135
        dd = {
            'serial_number': [],
            'angle': [],
            'rotation': [],
            'wave': [],
        }
        for n in range( len( files ) ):
            rot = int(files[n].split('_')[-4].strip("rot")) * step
            tmp,t = load_theta(files[n], NumTransducers=NumTransducers )
            data[:,:,n] = tmp
            for i, a in enumerate( angles ):
                dd['serial_number'].append(bc)
                dd['angle'].append(a)
                dd['rotation'].append(rot)
                dd['wave'].append(tmp[:,i])
        
        df = pd.DataFrame().from_dict(dd)
        signals.append(df)
    
    signals = pd.concat(signals)
    rotations  = list( np.array( [ [rot]*3 for rot in np.linspace( 0., 30., len( files ) ) ] ).ravel() )

    signals.rotation = rotations

    signals = easi_data.Signals(signals)
    signals.t = t / 1e6
    signals.angle = signals.angle.astype(int)
    return signals, files
    

    

    








    
    





    


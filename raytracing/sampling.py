###################################################################################################
# 
# sampling.py
# 
# Methods for sampling along lines and calculating time of 
# flight and other wave motion considerations 
# 
# Siddharth Maddali
# siddharth@liminalinsights.com
# Aug 2025
# 
###################################################################################################

import numpy as np

def SampleAlongLine( origin, direction, step_size, num_steps ):
    '''
    Sample points along a line defined by an origin and direction vector.
    '''
    assert np.ndim( origin ) ==np.ndim( direction ) == 1, 'Inputs should be 1D arrays. '
    direction /= np.linalg.norm(direction)
    sample_points = origin[:,np.newaxis].repeat( num_steps, axis=1 ) + step_size*direction[:,np.newaxis]*np.arange( num_steps )[np.newaxis,:]
    return sample_points

def EstimateTimeOfFlight( point_query, point_origin, speed_map, steps_per_pixel=20 ):
    '''
    Assumes that point_query and point_origin are already in pixel coordinates. 
    '''
    displacement = point_query - point_origin
    distance = np.linalg.norm( displacement )
    direction = displacement / distance
    num_steps = np.round( steps_per_pixel * distance ).astype( int )
    scale = np.linspace( 0., distance, num_steps )
    step_size = ( scale[1:] - scale[:-1] ).mean()
    sample_points = SampleAlongLine( origin=point_origin, direction=direction, step_size=step_size, num_steps=num_steps )
    sample_points = np.round( sample_points ).astype( int )
    speed = speed_map[ *sample_points ]
    time_per_step = step_size / speed # CAUTION: this step size is in pixel units. 
    total_time = time_per_step.sum()  # Multiply by physical step size to get physical time.   
    return total_time, speed 





    


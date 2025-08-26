################################################################################
# 
# smatrix.py
# 
# An ultrasound scattering matrix (S-matrix) module for various geometries. 
# Constructor builds transfer matrix for single layer with the correct 
# boundary conditions. Base class SMatrix designed for 1-D thru-wave 
# geometry at present. Support for more complicated geometries 
# coming soon. 
# 
# Siddharth Maddali
# siddharth@liminalinsights.com
# 
################################################################################

import numpy as np
from logzero import logger

class SMatrix: 

    def __init__( self, omega, length, speed, Z_triplet, time_scaler=1. ):
        assert Z_triplet.size==3, 'Should provide an acoustic impedance triplet when using constructor. '
        self.Z = Z_triplet
        self.omega = omega
        self.length = [ length ]
        self.speed = [ speed ]
        self.t0 = length / speed / time_scaler
        self.Build()

    def Build( self ):
        denominator = self.Z[1:] + self.Z[:2]
        t_forw = 2.*self.Z[1:] / denominator                    # transmission coefficient L to R
        t_back = 2.*self.Z[:2] / denominator                    # transmission coefficient R to L
        r_forw = ( self.Z[1:] - self.Z[:2] ) / denominator      # reflection coefficient L to R
        r_back = -r_forw                                        # reflection coefficient R to L


        phi = self.omega * self.t0
        w = np.exp( 1.j*phi ) # Effect of time delay in Fourier space
        all_internal_reflections = 1 - r_forw[1]*r_back[0]*( w**2 )
        S11 = ( np.prod( t_forw ) * w ) / all_internal_reflections
        S22 = ( np.prod( t_back ) * w ) / all_internal_reflections
        S12 = r_back[1] + ( t_back[1]*r_back[0]*t_forw[1] * (w**2) ) / all_internal_reflections
        S21 = r_forw[0] + ( t_forw[0]*r_forw[1]*t_back[0] * (w**2) ) / all_internal_reflections

        self.S = np.array(
            [ 
                [ S11, S12 ], 
                [ S21, S22 ]
            ]
        ) # the full scattering matrix
        return
    
    def __mul__( self, S2 ):
        assert ( self.Z[-2:]==S2.Z[:2] ).all(), 'Composing S-matrices requires matched impedances. '
        assert self.omega==S2.omega, 'Frequency mismatch. ' 
        Sout = SMatrix( self.omega, self.length, self.speed, self.Z )
        Sout.length.extend( S2.length )
        Sout.speed.extend( S2.speed )
        Sout.Z = np.concatenate( ( Sout.Z, S2.Z[2:] ) ) 
        Sout.S = Sout.S @ S2.S
        return Sout
    
    def __imul__( self, S2 ):
        assert ( self.Z[-2:]==S2.Z[:2] ).all(), 'Composing S-matrices requires matched impedances. '
        assert self.omega==S2.omega, 'Frequency mismatch. ' 
        self.length.extend( S2.length )
        self.speed.extend( S2.speed )
        self.Z = np.concatenate( ( self.Z, S2.Z[2:] ) ) 
        self.S = self.S @ S2.S
        return self








        

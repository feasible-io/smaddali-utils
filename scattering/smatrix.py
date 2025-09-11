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
import numba

import numpy as np
import functools as ftls

from logzero import logger


# def Redheffer( A, B ):
#     '''
#     Implementation of the Redheffer star product for two scattering matrices. 
#     Currently implemented only for 2x2 matrices. 
#     '''
#     A11, A12, A21, A22 = A[0,0], A[0,1], A[1,0], A[1,1]
#     B11, B12, B21, B22 = B[0,0], B[0,1], B[1,0], B[1,1]
#     denom = 1. - A12*B21
#     S11 = A11*B11/denom 
#     S12 = B12 + B11*B22*A12/denom
#     S21 = A21 + A11*A22*B21/denom
#     S22 = A22*B22/denom 

#     S = np.concatenate( 
#         ( 
#             np.array( [ S11, S12 ] )[np.newaxis,:], 
#             np.array( [ S21, S22 ] )[np.newaxis,:]
#         ), 
#         axis=0
#     )
#     return S

@numba.jit( nopython=True, cache=True )
def Redheffer(A, B):
    """
    Implementation of the Redheffer star product for two scattering matrices.
    Currently implemented only for 2x2 matrices.
    """
    # Unpack matrix elements directly, which is clean and efficient
    A11, A12, A21, A22 = A.ravel()
    B11, B12, B21, B22 = B.ravel()
    denom = 1.0 - A12 * B21

    # Pre-allocate the output matrix S with the correct shape and data type
    S = np.zeros((2, 2), dtype=A.dtype)

    # Calculate and assign elements directly to the pre-allocated array
    S[0,0] = A11 * B11 / denom
    S[0,1] = B12 + ( B11 * B22 * A12 / denom )
    S[1,0] = A21 + ( A11 * A22 * B21 / denom )
    S[1,1] = A22 * B22 / denom

    return S


class SMatrix: 

    def __init__( self, omega, length, speed, Z, field='velocity', time_offset=0. ):
        # assert Z.size==3, 'Should provide an acoustic impedance triplet when using constructor. '
        self.Z = Z
        self.offset = time_offset
        self.omega = omega
        assert field in [ 'pressure', 'velocity' ], 'Field must either be "pressure" or "velocity". '
        self.field = field
        simple_structure = not isinstance( length, list ) and not isinstance( speed, list )
        if simple_structure:  
            self.length = [ length, ]
            self.speed = [ speed, ]
            self.t0 = length / speed
            self.Build()
        else:
            self.length = length
            self.speed = speed
            # self.t0 = ftls.reduce( lambda x, y: x+y, [ l/v/self.scale for l, v in zip( length, speed ) ] )

    def Build( self ):
        '''
        Build the S-matrix elements based on the user-defined pressure or velocity field. 
        Note that these fields have different interfacial boundary conditions, and therefore 
        the corresponding reflection and transmission coefficients are different.
        '''
        denominator = self.Z[1:] + self.Z[:2]
        if self.field=='pressure':
            # these are coefficients for the pressure wave
            t_forw = 2.*self.Z[1:] / denominator                    # transmission coefficient L to R
            t_back = 2.*self.Z[:2] / denominator                    # transmission coefficient R to L
            r_forw = ( self.Z[1:] - self.Z[:2] ) / denominator      # reflection coefficient L to R
            r_back = -r_forw                                        # reflection coefficient R to L
        elif self.field=='velocity':
            # these are coefficients for the particle velocity wave
            t_forw = 2.*self.Z[:2] / denominator
            t_back = 2.*self.Z[1:] / denominator
            r_forw = ( self.Z[:2] - self.Z[1:] ) / denominator
            r_back = -r_forw
        self.phase_offset = np.exp( -1.j*self.omega*self.offset )
        self.phi = self.omega * self.t0
        w = np.exp( -1.j*self.phi ) # Effect of time delay in Fourier space
        # wconj = np.conj( w )
        rho = r_forw[1]*r_back[0]
        S11 = ( np.prod( t_forw ) * w ) / ( 1. - rho*(w**2) )
        S22 = ( np.prod( t_back ) * w ) / ( 1. - rho*(w**2) )
        S12 = r_back[1] + ( t_back[1]*r_back[0]*t_forw[1] * ( w**2 ) ) / ( 1. - rho*(w**2) )
        S21 = r_forw[0] + ( t_forw[0]*r_forw[1]*t_back[0] * ( w**2 ) ) / ( 1. - rho*(w**2) )
        self.S = self.phase_offset * np.array(
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
        Sout.S = Redheffer( self.S, S2.S )
        return Sout
    
    def __imul__( self, S2 ):
        assert ( self.Z[-2:]==S2.Z[:2] ).all(), 'Composing S-matrices requires matched impedances. '
        assert self.omega==S2.omega, 'Frequency mismatch. ' 
        self.length.extend( S2.length )
        self.speed.extend( S2.speed )
        self.Z = np.concatenate( ( self.Z, S2.Z[2:] ) ) 
        self.S = Redheffer( self.S, S2.S )
        return self
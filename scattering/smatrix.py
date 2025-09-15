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
import copy

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
    '''
    One-dimensional S-matrix class of waves. 
    '''

    def __init__( self, omega, Z, speed, x, field='pressure' ):
        '''
        THe S-matrix is defined for an interface located  at x, 
        with acoustic impedances given by the array Z (left to right), 
        and the frequency-dependent wavenumber k of the medium at the 
        observation point (x=0). 
        '''
        assert Z.size==2, 'Should provide an acoustic impedance doublet when using constructor. '
        assert x >= 0., 'Should provide a positive distance. '
        self.Z = Z
        self.omega = omega
        assert field in [ 'pressure', 'velocity' ], 'Field must be "pressure" or "velocity". '
        self.field = field
        self.speed = [ speed ]
        self.x = [ x ]
        self.Build()

    def Build( self ):
        '''
        Build the S-matrix elements for the parameterized interface. 
        '''
        denominator = self.Z.sum()
        T21, T12 = [ 2.*z/denominator for z in self.Z ]
        R12, R21 = [ sgn*( self.Z[1]-self.Z[0] )/denominator for sgn in [ 1., -1. ] ]
        if self.field=='velocity':
            T12, T21 = T21, T12
            R12, R21 = R21, R12
        phase = self.omega/self.speed[0] * self.x[0] # non-dispersive for now
        phi = np.exp( -1.j * phase )
        self.S = np.array( 
            [ 
                [ T12*phi, R21 ], 
                [ R12*phi*phi, T21*phi ]
            ]
        )

    def Copy( self ):
        return copy.deepcopy( self )

    def __mul__( self, S2 ):
        assert self.Z[1]==S2.Z[0], 'Composing S-matrices requires matched impedances. '
        assert self.omega==S2.omega, 'Frequency mismatch. ' 
        Sout = self.Copy()
        Sout.x.extend( S2.x )
        Sout.speed.extend( S2.speed )
        Sout.Z = np.concatenate( [ Sout.Z[:-1], S2.Z[1:] ] )
        Sout.S = Redheffer( self.S, S2.S )
        return Sout
    
    def __imul__( self, S2 ):
        assert self.Z[1]==S2.Z[0], 'Composing S-matrices requires matched impedances. '
        assert self.omega==S2.omega, 'Frequency mismatch. ' 
        self.x.extend( S2.x )
        self.speed.extend( S2.speed )
        self.Z = np.concatenate( [ self.Z[:-1], S2.z[1:] ] )
        self.S = Redheffer( self.S, S2.S )
        return self







    

        


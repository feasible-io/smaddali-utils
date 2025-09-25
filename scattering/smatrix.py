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
import sympy as sp

from logzero import logger
from tqdm.auto import tqdm


@numba.jit( nopython=True, cache=True )
def Redheffer(A, B):
    """
    Implementation of the Redheffer star product for two scattering matrices.
    Currently implemented only for 2x2 matrices.
    """
    A11, A12, A21, A22 = A.ravel()
    B11, B12, B21, B22 = B.ravel()
    denom = 1.0 - A12 * B21
    S = np.zeros((2, 2), dtype=A.dtype)
    S[0,0] = A11 * B11 / denom
    S[0,1] = B12 + ( B11 * B22 * A12 / denom )
    S[1,0] = A21 + ( A11 * A22 * B21 / denom )
    S[1,1] = A22 * B22 / denom
    return S

def RedhefferSymbolic( A, B ):
    denom = 1 - A[0,1]*B[1,0]
    Mred = sp.Matrix( 
        [ 
            [ A[0,0]*B[0,0]/denom, B[0,1] + ( B[0,0]*B[1,1]*A[0,1] ) / denom ], 
            [ A[1,0] + ( A[0,0]*A[1,1]*B[1,0] ) / denom, A[1,1]*B[1,1]/denom ]
        ]
    )
    return Mred


class SMatrix: 
    '''
    One-dimensional S-matrix class of waves. 
    '''

    def __init__( self, omega, Z, speed, x, absorption_coeff=0., field='pressure', build_symbolic=False ):
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
        self.abs_bb = [ absorption_coeff ] # broadband, frequency-independent absorption coefficient
        self.Build()
        if build_symbolic: 
            self.BuildFourierMatrixFunction()

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
        phase = ( self.omega/self.speed[0] + self.abs_bb[0] ) * self.x[0] # non-dispersive for now
        phi = np.exp( -1.j * phase ) # minus sign accounts for the sign effect in the wave equation
        self.S = np.array( 
            [ 
                [ T12*phi, R21 ], 
                [ R12*phi*phi, T21*phi ]
            ]
        )

    def Copy( self ):
        return copy.deepcopy( self )

    def __mul__( self, S2 ):
        assert self.Z[-1]==S2.Z[0], 'Composing S-matrices requires matched impedances. '
        assert self.omega==S2.omega, 'Frequency mismatch. ' 
        Sout = self.Copy()
        Sout.x.extend( S2.x )
        Sout.speed.extend( S2.speed )
        Sout.abs_bb.extend( S2.abs_bb )
        Sout.Z = np.concatenate( [ Sout.Z, S2.Z[1:] ] )
        Sout.S = Redheffer( self.S, S2.S )
        return Sout
    
    def __imul__( self, S2 ):
        assert self.Z[-1]==S2.Z[0], 'Composing S-matrices requires matched impedances. '
        assert self.omega==S2.omega, 'Frequency mismatch. ' 
        self.x.extend( S2.x )
        self.speed.extend( S2.speed )
        self.abs_bb.extend( S2.abs_bb )
        self.Z = np.concatenate( [ self.Z, S2.z[1:] ] )
        self.S = Redheffer( self.S, S2.S )
        return self
    
    def BuildSymbolic( self, cse=False, modules='numpy' ):
        logger.warning( 'Absorption not yet implemented in symbolic matrix. ' )
       
        # define all coefficients
        T_forw = sp.symbols( ' '.join( [ f'T{n}{n+1}' for n in range( self.Z.size-1 ) ] ) )
        T_back = sp.symbols( ' '.join( [ f'T{n+1}{n}' for n in range( self.Z.size-1 ) ] ) )
        R_forw = sp.symbols( ' '.join( [ f'R{n}{n+1}' for n in range( self.Z.size-1 ) ] ) )
        R_back = sp.symbols( ' '.join( [ f'R{n+1}{n}' for n in range( self.Z.size-1 ) ] ) )
        time = sp.symbols(  ' '.join( [f't{n}' for n in range( len( self.speed ) )] ) ) # this is L/v for each material
        omega = sp.symbols( 'omega' )
        self.syms = ( omega, ) + time + T_forw + T_back + R_forw + R_back
        smatrix_list = []
        for n in range( len( time ) ):
            my_exp = sp.exp( -sp.I * omega * time[n] )
            M = sp.Matrix( 
                [ 
                    [ T_forw[n]*my_exp, R_back[n] ], 
                    [ R_forw[n]*my_exp*my_exp, T_back[n]*my_exp ]
                ]
            )
            smatrix_list.append( sp.simplify( M ) )
        self.M = ftls.reduce( RedhefferSymbolic, tqdm( smatrix_list, desc='Composing S-matrices' ) )
        if cse: 
            self.M_replacement_00, self.M_reduced_00 = sp.cse( self.M[0,0], optimizations='basic' ) 
            self.replacement_funcs = []
            self.replacement_args = []
            for _, expr in self.M_replacement_00:
                lst = list( expr.free_symbols )
                lst.sort( key=lambda x: str( x ) )
                lst = tuple( lst )
                self.replacement_args.append( tuple(  str( l ) for l in lst ) ) # to be used as dict keys at evaluation time
                myfun = sp.lambdify( lst, expr, modules=modules )
                self.replacement_funcs.append( myfun )
            lst = list( self.M_reduced_00[0].free_symbols )
            lst.sort( key=lambda x: str( x ) )
            self.args_numerical_final = tuple( str( l ) for l in lst )
            self.transfer_function = sp.lambdify( tuple( lst ), self.M_reduced_00[0], modules=modules )
        return
    
    def Evaluate( self, **input_args ):
        '''
        Returns optimized numerical evaluation of S-matrix symbolic expression, 
        given the physical parameters as input. 
        '''
        assert hasattr( self, 'replacement_funcs' ), 'Intermediate sub-expressions not generated. Please rerun BuildSymbolic with "cse=True". '
        for var, args, func in zip( self.M_replacement_00, self.replacement_args, self.replacement_funcs ):
            args_numerical = tuple( input_args[st] for st in args )
            input_args[ str( var ) ] = func( *args_numerical )
        value = self.transfer_function( *( self.args_numerical_final ) )
        return value 
            





        
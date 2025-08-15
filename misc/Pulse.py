########################################################################
# Pulse.py
# 1-D Ultrasound pulse generator routines for FDTD simulations. 
# 
# Siddharth Maddali
# Dec 2024
########################################################################

import numpy as np
import sympy as spy
from logzero import logger
from numpy.fft import ifftshift, fftn, ifftn

def CesaroCorrection(f):
    """
    Apply Cesaro correction to a given function.

    Parameters:
    f (ndarray): Input array representing the function.

    Returns:
    tuple: Corrected function and its Fourier transform.
    """
    fhat = fftn(f, norm='ortho')
    N = f.size // 2  # assuming even-sized array
    orders = ifftshift(np.arange(-N, N))
    correction = (N + 1 - np.abs(orders)) / N
    fhat *= correction
    f_corrected = ifftn(fhat, norm='ortho')
    return f_corrected, fhat

def pulse_smoothed(th, N, th0):
    """
    Generate a smoothed pulse using recursive Fourier series.

    Parameters:
    th (ndarray): Grid of theta values.
    N (int): Order of the Fourier series.
    th0 (float): Initial theta value.

    Returns:
    tuple: Fourier series and Fejer kernel.
    """
    if N == 0:
        return tuple(th0 * np.ones(th.shape) for _ in range(2))
    else:
        this_fourier = (np.sin(N * th0 / 2) / (np.pi * N)) * np.exp(1.j * N * th) / np.sqrt(2 * np.pi)
        partial_fourier, partial_fejer = pulse_smoothed(th, N - 1, th0)
        return this_fourier + partial_fourier, ((N - 1) * partial_fejer + partial_fourier) / N

def Square(grid, location, width, edge_type='cosine'):
    """
    Generate a square pulse with different edge types.

    Parameters:
    grid (ndarray): 1-D array representing the grid.
    location (float): Center of the pulse in the grid.
    width (float): Width of the square pulse in grid units.
    edge_type (str): Type of edge ('discontinuous', 'cosine', 'cesaro', 'gibbs').

    Returns:
    ndarray: Array representing the square pulse.
    """
    if edge_type == 'discontinuous':
        return (np.abs(grid - location) < width / 2).astype(float)
    elif edge_type == 'cosine':
        cosine_width = width / 20
        pulse = np.zeros(grid.shape)
        
        # Left edge
        here_left = np.where(np.abs(grid - (location - width / 2)) < cosine_width / 2)
        grid_vals_left = grid[here_left]
        left_edge = np.sin(-np.pi / 2 + np.pi * (grid_vals_left - grid_vals_left.min()) / (grid_vals_left.max() - grid_vals_left.min()))
        pulse[here_left] = 0.5 * (1 + left_edge)
        
        # Right edge
        here_right = np.where(np.abs(grid - (location + width / 2)) < cosine_width / 2)
        grid_vals_right = grid[here_right]
        right_edge = np.sin(np.pi / 2 + np.pi * (grid_vals_right - grid_vals_right.min()) / (grid_vals_right.max() - grid_vals_right.min()))
        pulse[here_right] = 0.5 * (1 + right_edge)
        
        # Center
        pulse[np.where(np.abs(grid - location) < (width - cosine_width) / 2)] = 1
        return pulse
    elif edge_type in ['cesaro', 'gibbs']:
        choose = 1 if edge_type == 'cesaro' else 0
        grid_shifted = grid - location
        left_extent = grid_shifted[0]
        left_extent *= np.sign(left_extent)
        num = (np.abs(grid_shifted) < left_extent).sum()
        num += (num % 2)
        theta_grid = np.linspace(-np.pi, np.pi, num)
        theta_width = np.pi * width / np.abs(left_extent)
        pulse = np.zeros(grid.shape)
        pulse_section = np.real(pulse_smoothed(theta_grid, 1000, theta_width)[choose])
        baseline = (pulse_section[0] + pulse_section[-1]) / 2
        pulse_section -= baseline
        pulse_section /= pulse_section.max()
        pulse[:theta_grid.size] = pulse_section
        return pulse
    else:
        logger.error('Unrecognized edge type. Returning discontinuous edge.')
        return (np.abs(grid - location) < width / 2).astype(float)

def HermiteGauss(grid, location, spread, order):
    """
    Generate a Hermite-Gauss function.

    Parameters:
    grid (ndarray): 1-D array representing the grid.
    location (float): Center of the function in the grid.
    spread (float): Spread of the Gaussian.
    order (int): Order of the Hermite polynomial.

    Returns:
    ndarray: Array representing the Hermite-Gauss function.
    """
    assert order >= 0, 'Hermite-Gauss index should be >= 0'
    arg = (grid - location) / spread
    if order == 0:
        return np.exp(-arg**2)  # unnormalized Gaussian
    elif order == 1:
        x = spy.symbols('x')
        H1sym = spy.lambdify(x, -spy.exp(-x**2 / 2) * spy.diff(spy.exp(-x**2 / 2), x))  # symbolic expression for H1 for easy differentiation
        return H1sym(arg)
    else:
        return arg * HermiteGauss(grid, location, spread, order - 1) - np.sqrt(order - 1) * HermiteGauss(grid, location, spread, order - 2)

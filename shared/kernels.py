import numpy as np

def uniform_kernel(u):
    return 0.5 * (np.abs(u) <= 1.0)

def triangular_kernel(u):
    return (1.0 - np.abs(u)) * (np.abs(u) <= 1.0)

def epanechnikov_kernel(u):
    return 0.75 * (1.0 - u**2) * (np.abs(u) <= 1.0)

def quartic_kernel(u):
    """Квадратичное (бивесовое) ядро, может использоваться в LOWESS"""
    return (15.0 / 16.0) * ((1.0 - u**2)**2) * (np.abs(u) <= 1.0)

def gaussian_kernel(u):
    return (1.0 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * u**2)

KERNELS = {
    'uniform': uniform_kernel,
    'triangular': triangular_kernel,
    'epanechnikov': epanechnikov_kernel,
    'quartic': quartic_kernel,
    'gaussian': gaussian_kernel
}
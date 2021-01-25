import numpy as np
from numpy.polynomial.legendre import leggauss


class TimeFunc:

    def __init__(self, t_values, y_values):
        self.t_values = t_values
        self.y_values = y_values

    def __call__(self, t):
        if isinstance(t, np.ndarray):  # array
            return np.array([self.y_values[np.argmin(np.abs(self.t_values - tt))] for tt in t])
        else:  # single number
            idx = np.argmin(np.abs(self.t_values - t))
            return self.y_values[idx]


def compute_conv_at_t(f, g, t, f_bounds, g_bounds, n_quad):
    """
    Compute the convolution at time `t` between the two function `f` and `g`
    with bounded support `f_bounds` and `g_bounds` respectively, using the
    Gauss-Legendre quadrature method with `n_quad` points.

    Arguments
    ---------
    f : callable
        Function taking as input a one-dimensional numpy.ndarray
    g : callable
        Function taking as input a one-dimensional numpy.ndarray
    t : float
        Query time
    f_bounds : tuple
        Tuple of length 2 with min and max support value respectively
    g_bounds : tuple
        Tuple of length 2 with min and max support value respectively
    n_quad : int
        Number of Gauss-Legendre quadrature points
    """
    f_supp_len = f_bounds[1] - f_bounds[0]
    g_supp_len = g_bounds[1] - g_bounds[0]

    xx, ww = leggauss(n_quad)

    if g_supp_len < f_supp_len:  # integrate on support of g
        xx = (xx + 1) * (g_bounds[1] - g_bounds[0]) / 2 + g_bounds[0]
        ww = ww * (g_bounds[1] - g_bounds[0]) / 2
        val_at_t = np.sum(ww * g(xx) * f(t - xx))
        return val_at_t

    else:  # integrate on support of f
        xx = (xx + 1) * (f_bounds[1] - f_bounds[0]) / 2 + f_bounds[0]
        ww = ww * (f_bounds[1] - f_bounds[0]) / 2
        val_at_t = np.sum(ww * g(t - xx) * f(xx))
        return val_at_t


def compute_integrated_time_func(time_func):
    """
    Compute the integral of the function `func` at `n_points` with
    the `bounds`
    """
    t_values = time_func.t_values
    y_values = np.zeros_like(t_values)
    for i, t in enumerate(t_values):
        idx = np.argmin(np.abs(time_func.t_values - t))
        if idx > 1:
            y_values[i] = np.sum(time_func.y_values[1:idx] * np.diff(time_func.t_values[:idx]))
    return TimeFunc(t_values, y_values)


def compute_G_tilde_func(kernel, noise_pdf, kernel_support, noise_support_width, n_quad, n_points):
    M = noise_support_width
    A = kernel_support

    # If zero-kernel, then return zero-kernel
    if np.allclose(kernel(np.linspace(0, A, n_points)), 0.0):
        return TimeFunc(np.array([0.0, 1.0]), np.array([0.0, 0.0]))

    # Compute (f_i * g_ij)(t) for all t and build callable interpolation function
    t_values_1 = np.linspace(-M/2, A + M/2, n_points)
    y_values_1 = np.array([compute_conv_at_t(f=noise_pdf,
                                    g=kernel,
                                    t=t,
                                    f_bounds=[-M/2, M/2],
                                    g_bounds=[0.0, A],
                                    n_quad=n_quad)
                  for t in t_values_1])
    f_i_conv_g_ij = TimeFunc(t_values_1, y_values_1)

    t_values_2 = np.linspace(-M, A + M, n_points)
    y_values_2 = np.array([compute_conv_at_t(f=noise_pdf,
                                    g=f_i_conv_g_ij,
                                    t=t,
                                    f_bounds=[-M/2, M/2],
                                    g_bounds=[-M/2, A + M/2],
                                    n_quad=n_quad)
                  for t in t_values_2])
    g_tilde_ij = TimeFunc(t_values_2, y_values_2)

    return g_tilde_ij

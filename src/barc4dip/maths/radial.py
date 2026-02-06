# SPDX-License-Identifier: CECILL-2.1
# Copyright (c) 2026 ESRF - the European Synchrotron

from __future__ import annotations


def calc_radial_average(signal_2d: np.ndarray, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate the radial average of a 2D signal.

    Args:
        signal_2d (np.ndarray): 2D array representing the signal to be radially averaged.
        x (np.ndarray): 1D array of x-axis coordinates.
        y (np.ndarray): 1D array of y-axis coordinates.

    Returns:
        Tuple[np.ndarray, np.ndarray]: 
            - Radially averaged values as a 1D array.
            - Corresponding radial distances as a 1D array.
    """
    def _polar_to_cartesian(theta: np.ndarray, rho: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        x = rho * np.cos(theta)
        y = rho * np.sin(theta)
        return x, y

    x_start, x_end, nx = x[0], x[-1], x.size
    y_start, y_end, ny = y[0], y[-1], y.size
    x_center = 0.5 * (x_end + x_start)
    y_center = 0.5 * (y_end + y_start)

    range_r = [0, min(x_end - x_center, y_end - y_center)]
    range_theta = [0, 2 * np.pi]

    nr = int(nx * 0.5)
    ntheta = int((range_theta[1] - range_theta[0]) * 180)

    X = np.linspace(x_start, x_end, nx)
    Y = np.linspace(y_start, y_end, ny)

    R = np.linspace(range_r[0], range_r[1], nr)
    THETA = np.linspace(range_theta[0], range_theta[1], ntheta)

    R_grid, THETA_grid = np.meshgrid(R, THETA, indexing='ij')
    X_grid, Y_grid = _polar_to_cartesian(THETA_grid, R_grid)

    interp_func = RegularGridInterpolator((y, x), signal_2d, bounds_error=False, fill_value=0)

    points = np.column_stack([Y_grid.ravel(), X_grid.ravel()])
    values = interp_func(points).reshape(R_grid.shape)

    radial_avg = np.mean(values, axis=1)

    return radial_avg, R
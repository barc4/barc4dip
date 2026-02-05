# SPDX-License-Identifier: CECILL-2.1
# Copyright (c) 2025 Synchrotron SOLEIL

"""
helper for measuring and printing elapsed wall-clock time.
"""

from __future__ import annotations

from time import time

def now():
    """
    Return the current wall-clock time.

    Returns:
        float: Current time in seconds since the epoch, as returned by time().
    """
    return time()

def elapsed_time(t_start: float, verbose: bool = True) -> float:
    """
    Compute and optionally print the elapsed wall-clock time.

    Parameters:
        t_start (float): Reference start time in seconds.
        verbose (bool): If True, print the formatted elapsed time. If False,
            only return the elapsed time. Default is True.

    Returns:
        float: Elapsed time in seconds since t_start.
    """
    delta_t = time() - t_start

    if verbose:
        if delta_t < 1.0:
            print(f">> Total elapsed time: {delta_t * 1000.0:.2f} ms")
            return

        hours, rem = divmod(delta_t, 3600.0)
        minutes, seconds = divmod(rem, 60.0)

        if hours >= 1.0:
            print(
                f">> Total elapsed time: "
                f"{int(hours)} h {int(minutes)} min {seconds:.2f} s"
            )
        elif minutes >= 1.0:
            print(f">> Total elapsed time: {int(minutes)} min {seconds:.2f} s")
        else:
            print(f">> Total elapsed time: {seconds:.2f} s")

    return delta_t
        
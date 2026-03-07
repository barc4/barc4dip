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
        

def progress_update(loop_name: str, t: int, T: int, last_bucket: int) -> int:
    """
    Print a simple textual progress bar for a loop.

    The progress is quantized into 10 buckets (0-100% in steps of 10%).
    The function prints an updated progress bar only when the bucket
    changes, avoiding excessive terminal output.

    Parameters:
        loop_name (str):
            Name of the loop displayed before the progress bar.
        t (int):
            Current loop index (typically the frame or iteration number).
        T (int):
            Total number of iterations in the loop.
        last_bucket (int):
            Previously printed progress bucket (0-10). Used to detect
            whether the bar should be updated.

    Returns:
        int:
            Updated progress bucket (0-10). This value should be stored
            and passed back to the function on the next iteration.
    """
    bucket = int((10 * t) // max(1, T - 1)) 
    if bucket != last_bucket:
        progress = 10 * bucket
        bar = "#" * bucket + "-" * (10 - bucket)
        print(f"\r{loop_name}: [{bar}] {progress:3d}%", end="", flush=True)
        return bucket
    return last_bucket

def progress_done(loop_name: str) -> None:
    """
    Print the final 100% completion line for a progress bar.

    This function should be called after a loop that used
    :func:`progress_update`, ensuring the terminal output ends
    with a completed progress bar.

    Parameters:
        loop_name (str):
            Name of the loop displayed before the progress bar.

    Returns:
        None
    """
    print(f"\r{loop_name}: [##########] 100%", flush=True)
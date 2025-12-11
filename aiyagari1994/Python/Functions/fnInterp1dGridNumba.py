from numba import njit
import numpy as np

@njit
def fnInterp1dGridNumba(
        x: float, 
        grid: np.ndarray, 
        Ngrid: float, 
        ) -> tuple:
    """
    Interpolate x on a Grid using Numba @njit for faster computation
    """
    x = np.asarray(x)
    low = np.searchsorted(grid, x, side='right') - 1
    low = max(0, min(int(Ngrid) - 2, low))
    high = low + 1
    wtLow = (grid[high] - x) / (grid[high] - grid[low])
    wtLow = max(0.0, min(1.0, wtLow))
    wtHigh = 1 - wtLow
    return low, high, wtLow, wtHigh
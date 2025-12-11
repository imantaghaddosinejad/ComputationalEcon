import numpy as np

def fnInterp1dGrid(
        x: float, 
        grid: np.ndarray, 
        Ngrid: float, 
        ) -> tuple:
    """
    Interpolate x on a Grid
    
    Args:
        x: Value to interpolate
        grid: Grid to interpolate on
        Ngrid: Number of Grid Points
    
    Returns:
        tuple: low, high, wtLow, wtHigh
    
    Raises:
        TypeError: If inputs don't match expected types
    """
    if not isinstance(x, (int, float, np.ndarray)):
        raise TypeError("x must be an int, float, or numpy array")
    if not isinstance(grid, np.ndarray):
        raise TypeError("grid must be a numpy array")
    if not isinstance(Ngrid, (int, float)):
        raise TypeError("Ngrid must be an int or float")
    
    x = np.asarray(x)
    low = np.searchsorted(grid, x, side='right') - 1
    low = np.maximum(0, np.minimum(Ngrid - 2, low))
    high = low + 1

    wtLow = (grid[high] - x) / (grid[high] - grid[low])
    wtLow = np.clip(wtLow, 0, 1)
    wtHigh = 1 - wtLow
    
    return low, high, wtLow, wtHigh
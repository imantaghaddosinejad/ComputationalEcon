import numpy as np

def fnHHobj(
        x: float, 
        beta: float, 
        budget: float, 
        grida: np.ndarray, 
        Na: float, 
        expvf: np.ndarray
        ) -> float:
    """
    Objective Function With Interpolation for Wealth

    Args:
        x: Value of aprime (a') chosen wealth
        beta: Discount factor
        budget: Budget constraint
        grida: Wealth grid
        Na: Number of grid points for wealth grid 
        expvf: Expected value of value function defined over wealth grid

    Returns:
        float: Value of the objective function
    """      
    aprime = x.item() if np.ndim(x) > 0 else x 
    c = np.maximum(1e-8, budget - aprime)

    aLow = np.searchsorted(grida, aprime, side='right') - 1 
    aLow = np.maximum(0, np.minimum(Na - 2, aLow))
    aHigh = aLow + 1

    wtLow = (grida[aHigh] - aprime) / (grida[aHigh] - grida[aLow])
    wtLow = np.clip(wtLow, 0, 1)
    wtHigh = 1 - wtLow  

    expval = wtLow*expvf[aLow] + wtHigh*expvf[aHigh]

    val = -(np.log(c) + beta * expval)
    return val
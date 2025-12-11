from typing import Tuple
import numpy as np
from scipy.optimize import minimize_scalar # type: ignore
from .fnHHobj import fnHHobj

def fnOptaprime(
        objargs: Tuple,
        amin: float,
        amax: float
        ) -> float:
    """
    Minimize Scalar Function With Constraints

    Args: 
        objargs: Tuple of arguments for fnHHobj (beta, budget, grida, Na, expvf)
        amin: Lower bound for the minimization
        amax: Upper bound for the minimization

    Returns:
        float: Optimal value that minimizes objective function     
    """    
    result = minimize_scalar(
        lambda x: fnHHobj(x, *objargs), 
        bounds=[amin, amax], 
        method='bounded', 
        options={'xatol': 1e-12, 'maxiter': 2000}
    )
    return result.x
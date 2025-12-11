import numpy as np
def fnMCss(
        mT: np.ndarray, 
        eigvlmethod: bool = True
        ) -> np.ndarray:
    """
    Compute the Stationary Distribution of a Markov Chain

    Args:
        mT: Transition Matrix (should be transposed)
        eigvlmethod: Use Eigenvalue Method (default) or Iterative Method    
    
    Returns:
        np.ndarray: Stationary Distribution of the Markov Chain

    Raises:
        TypeError: If inputs don't match expected types
    """
    if not isinstance(mT, np.ndarray):
        raise TypeError("mT must be a numpy array")
    if not isinstance(eigvlmethod, bool):
        raise TypeError("eigvlmethod must be a boolean")

    if eigvlmethod:
        _, eigvc = np.linalg.eig(mT)
        vec = eigvc[:, 0]
        vec = vec / np.sum(vec)
        return vec 
    else:
        err = 10 
        x0 = np.zeros(len(mT))
        x0[0] = 1
        while err >= 1e-12: 
            x1 = np.dot(mT, np.transpose(x0))
            err = np.max(np.abs(x1 - x0))
            x0 = np.copy(x1)
        return x0
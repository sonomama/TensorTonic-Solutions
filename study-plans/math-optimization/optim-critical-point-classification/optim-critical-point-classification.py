import numpy as np

def classify_critical_point(H: list) -> str:
    """
    Returns the second-order classification string.
    """
    eigvals = np.linalg.eigvalsh(np.array(H))
    tol = 0.000001 
    degenerate_check = (abs(eigvals) <= tol)
    if np.sum(degenerate_check):
        return "degenerate"
    is_pos = eigvals > tol
    counts = np.sum(is_pos)
    size = np.shape(eigvals)[0]
    if counts == size:
        return "local_min"
    elif counts == 0:
        return "local_max"
    return "saddle"
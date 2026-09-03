import numpy as np

def YJinv(y, lam):
    """
    Inverse Yeo-Johnson transformation.

    Parameters
    ----------
    y : array_like
        Transformed value(s).
    lam : float
        Yeo-Johnson parameter. Must not be 0 or 2.

    Returns
    -------
    x : ndarray or scalar
        Original value(s).
    """
    if lam == 0 or lam == 2:
        raise ValueError("This function requires lambda != 0 and lambda != 2.")

    y = np.asarray(y, dtype=float)

    x = np.empty_like(y)

    pos = y >= 0
    neg = ~pos

    # y >= 0:
    # y = ((x + 1)^lambda - 1) / lambda
    x[pos] = np.power(1 + lam * y[pos], 1 / lam) - 1

    # y < 0:
    # y = -[((1 - x)^(2-lambda) - 1) / (2-lambda)]
    x[neg] = 1 - np.power(
        1 - (2 - lam) * y[neg],
        1 / (2 - lam)
    )

    return x
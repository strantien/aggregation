import numpy as np

XMIN, XMAX = -1, 1


def rho0(x, J, xmin: float = XMIN, xmax: float = XMAX):
    """
    Initial condition
    """
    L = xmax - xmin
    xl = xmin + 0.25 * L
    xr = xmin + 0.75 * L
    ## One bump
    return np.exp(-20.0 * (x - (xmin + xmax) / 2) ** 2)
    ## Two bumps
    # return(np.exp(-40.*(x - xl)**2) + np.exp(-40.*(x - xr)**2))
    ## Two peaked bumps far away oen from another
    # return(np.exp(-200.*(x + 0.2*L)**2) + np.exp(-200.*(x - 0.2*L)**2))
    # return ((x - (xmin + xmax)/2 > -0.25)*(x - (xmin + xmax)/2 < 0.25)).astype(float)
    ## A 'rectangular' Dirac mass
    # return((x == (xmin+xmax)/2).astype(float))
    ## A single Dirac mass
    # return((x == xmin + (int((2*J+1)/4) + 0.5) * L/(2*J + 1)).astype(float) + (x == xmin + (int(3*(2*J+1)/4) + 0.5) * L/(2*J + 1)).astype(float))
    ## Two Dirac masses


def fgradW(x, alpha: float = 1.0, x0: float = 0.5):
    """
    Gradient of the potential
    """
    ## Gradient of |x|^{alpha + 1}
    # return(np.sign(x) * (np.abs(x) ** alpha))
    ## Same but linearly cut for |x| > x0
    return np.sign(x) * (
        (np.abs(x) ** (alpha + 1)) * (np.abs(x) <= x0)
        + ((alpha + 1) * x0**alpha * (np.abs(x) - x0) + x0 ** (alpha + 1))
        * (np.abs(x) > x0)
    )

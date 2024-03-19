import ot


def wasserstein(rho1, rho2, X1, X2, p: float = 2):
    """
    Computes the p-Wasserstein distance between two densities rho1 and rho2, with respective support X1 and X2.
    """
    return ot.emd2_1d(X1, X2, rho1, rho2, metric="minkowski", p=p) ** (1 / p)

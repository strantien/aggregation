"""Simulation des solutions de l'équation d'agrégation-diffusion"""

from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import scipy.signal
import scipy.sparse.linalg as scspl
import seaborn as sns
from scipy.sparse import diags

# To do: pass rho0 as a parameter of Callable type
from aggdiff.parameters import rho0

XMIN, XMAX = -1, 1


class Solver:

    def __init__(
        self,
        eps: float,
        fgradW: Callable,
        is_repulsive: bool = False,
        xmin: float = XMIN,
        xmax: float = XMAX,
        scheme: str = "theta-scheme",
        theta: float = 1.0,
        nbC: float = 0.9,
        c_rusanov: float = 1.1,
        BC: str = "mass conservation",
    ):
        self.eps = eps
        self.fgradW = fgradW
        self.is_repulsive = is_repulsive
        self.xmin = xmin
        self.xmax = xmax
        self.scheme = scheme
        self.theta = theta
        self.c_rusanov = c_rusanov
        self.nbC = nbC
        self.BC = BC

    def compute_velocity(self, rho, kernel):
        """
        Basically computes a = rho convoluted with kernel
        """
        return -sp.signal.convolve(rho, kernel, mode="same", method="fft")

    def compute_time_step(self, rho, a, dx, dt, t, T, I, Laplacian):
        """
        From rho^n and a^n computes rho^{n+1}
        """
        vel_pos = a > 0
        # Warning: below is the classical convention for the negative part
        vel_neg = a <= 0

        arho = a * rho

        if self.scheme in ["theta-scheme", "upwind"]:
            aInfty = np.amax(np.fabs(a))
            dt = self.nbC / ((aInfty / dx) + (2 * self.eps * (1 - self.theta) / dx**2))
        elif self.scheme in ["rusanov", "lax-friedrichs"]:
            dt = self.nbC * dx / self.c_rusanov

        dt = min(dt, T - t)
        rhs = np.zeros_like(rho)

        if self.scheme in ["theta-scheme", "upwind"]:
            rhs[1:-1] -= (dt / dx) * np.fabs(a[1:-1]) * rho[1:-1]
            rhs[1:-1] -= (dt / dx) * arho[2:] * vel_neg[2:]
            rhs[1:-1] += (dt / dx) * arho[:-2] * vel_pos[:-2]
        # To do: check the boundary conditions for the Rusanov and Lax-Friedrichs schemes + check the Lax-Friedrichs scheme, why the hell is there a c_rusanov there
        elif self.scheme == "rusanov":
            rhs[1:-1] -= 0.5 * (arho[2:] - arho[:-2])
            rhs[1:-1] += 0.5 * self.c_rusanov * (rho[2:] - 2 * rho[1:-1] + rho[:-2])
        elif self.scheme == "lax-friedrichs":
            rhs[1:-1] = 0.25 * (
                -(a[1:-1] + a[2:]) * (rho[1:-1] + rho[2:])
                + (a[:-2] + a[1:-1]) * (rho[:-2] + rho[1:-1])
            )
            rhs[1:-1] += 0.5 * self.c_rusanov * (rho[2:] - 2 * rho[1:-1] + rho[:-2])

        if self.BC == "periodic":
            rhs[-1] -= (dt / dx) * (np.fabs(a[-1]) * rho[-1] + arho[0] * vel_neg[0])
            rhs[0] += (dt / dx) * (arho[-1] * vel_pos[-1] - np.fabs(a[0]) * rho[0])
        elif self.BC == "mass conservation":
            rhs[0] -= (dt / dx) * (arho[0] * vel_pos[0] + arho[1] * vel_neg[1])
            rhs[-1] += (dt / dx) * (arho[-2] * vel_pos[-2] + arho[-1] * vel_neg[-1])

        rho += rhs

        if self.eps > 0:
            rho[:] = scspl.spsolve(
                I - self.eps * self.theta * dt * Laplacian,
                scipy.sparse.csr_matrix.dot(
                    I + self.eps * (1 - self.theta) * dt * Laplacian, rho
                ),
            )

        return dt

    def compute_solution(
        self,
        J: int,
        T: float = 1.0,
        plot=False,
    ):
        L = self.xmax - self.xmin
        dx = L / (2 * J + 1)
        X = self.xmin + (np.arange(2 * J + 1) + 0.5) * dx
        t = 0
        dt = 1
        rho = rho0(X, J, self.xmin, self.xmax)
        rho = rho / np.sum(rho)
        rhoIni = rho.copy()
        a = np.zeros(2 * J + 1)

        I = diags([np.ones(2 * J + 1)], [0], format="csr")
        Laplacian = diags(
            [np.ones(2 * J), -2 * np.ones(2 * J + 1), np.ones(2 * J)],
            [-1, 0, 1],
            format="csr",
        )
        if self.BC == "periodic":
            Laplacian[0, 2 * J], Laplacian[2 * J, 0] = 1, 1
        elif self.BC == "mass conservation":
            Laplacian[0, 0], Laplacian[2 * J, 2 * J] = -1, -1
        Laplacian /= dx**2

        XgradW = np.arange(-(2 * J), 2 * J + 1) * dx
        gradW = self.fgradW(XgradW)
        gradW[2 * J] = 0
        if self.is_repulsive:
            gradW *= -1

        while t < T:
            a = self.compute_velocity(rho, gradW)
            dt = self.compute_time_step(rho, a, dx, dt, t, T, I, Laplacian)
            t += dt

        if plot:
            sns.set_style("whitegrid")
            palette = sns.color_palette("deep")
            fig = plt.figure()
            fig.set_figheight(8)
            fig.set_figwidth(11)
            ax1 = plt.subplot(211)
            ax2 = plt.subplot(212)
            ax1.set_title(rf"$\rho(T = {T:.2f})$")
            ax1.plot(
                X,
                rhoIni,
                linewidth=0.9,
                color=palette[0],
                linestyle=":",
                label=r"$\rho^{ini}$",
            )
            ax1.plot(X, rho, linewidth=0.9, color=palette[0], label=r"$\rho$")
            ax1.legend()
            ax1.grid(True)
            ax2.set_title(rf"$a[\rho](T = {T:.2f})$")
            ax2.plot(X, a, linewidth=0.9, color=palette[1])
            ax2.grid(True)
            fig.tight_layout()
            plt.show()

        return rho, a

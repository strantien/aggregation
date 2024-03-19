"""Simulation des solutions de l'équation d'agrégation-diffusion"""

import logging
import os.path
import pathlib
from multiprocessing import Pool

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import scipy.signal
import scipy.sparse.linalg as scspl
import seaborn as sns
from scipy.sparse import diags

from aggdiff.parameters import fgradW, rho0
from aggdiff.utils import wasserstein

REPO_DIR = os.path.dirname(os.path.dirname((os.path.realpath(__file__))))
TESTS_DIR = "tests/"

XMIN, XMAX = -1, 1


class Solver:

    def __init__(
        self,
        eps: float,
        scheme: str = "theta-scheme",
        theta: float = 1.0,
        nbC: float = 0.9,
        c_rusanov: float = 1.1,
        BC: str = "mass conservation",
    ):
        self.eps = eps
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

    def run_simulation(
        self,
        J: int,
        xmin: float = XMIN,
        xmax: float = XMAX,
        alpha: float = 1.0,
        x0: float = 0.5,
        T: float = 1.0,
        is_repulsive: bool = False,
        plot=False,
    ):
        L = xmax - xmin
        dx = L / (2 * J + 1)
        X = xmin + (np.arange(2 * J + 1) + 0.5) * dx
        t = 0
        dt = 1
        rho = rho0(X, J, xmin, xmax)
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
        gradW = fgradW(XgradW, alpha=alpha, x0=x0)
        gradW[2 * J] = 0
        if is_repulsive:
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


if __name__ == "__main__":
    run_simu = "y" in input("Run a simulation ? (y/n) ")
    if run_simu:
        eps = 0.05
        scheme = "theta-scheme"
        ## CFL number
        nbC = 0.9
        ## Constant in the Rusanov scheme
        c_rusanov = 1.1
        ## theta-scheme (implicit: theta = 1, explicit: theta = 0, Crank-Nicolson: theta = 0.5)
        theta = 1.0
        T = 0.5
        J = 5000
        solver = Solver(eps=eps, scheme=scheme, theta=theta, nbC=nbC)
        solver.run_simulation(J=J, T=T, plot=True)
    else:
        run_cvg = "y" in input("Draw convergence curves ? (y/n)")
        if run_cvg:
            cvg_param_name = input(
                "Convergence with respect to which parameter (dx or eps) ? "
            )
            if cvg_param_name == "dx":

                eps = 0.1
                xmin, xmax = XMIN, XMAX
                L = xmax - xmin
                alpha = 1.0
                x0 = 0.5
                p = 2
                T = 0.5
                nbC = 0.9
                scheme = "upwind"
                savedir = os.path.join(
                    REPO_DIR,
                    TESTS_DIR,
                    "aggdiff_convergence_evolutif_"
                    + scheme
                    + f"_diffImplicite_wrt_dx_eps_{eps}_alpha_{alpha}_p_{p}_T_{T}_CFL_{nbC}/",
                )
                if os.path.exists(savedir):
                    run = input(
                        "The directory already exists. Running the script will erase the previous simulations. Do you want to run anyway? (y/n)"
                    )
                    if "n" in run:
                        exit()
                pathlib.Path(savedir).mkdir(parents=True, exist_ok=True)
                list_J = [100 * 2**i for i in range(7)]
                solver = Solver(eps=eps, scheme=scheme, nbC=nbC)
                nproc = 2
                with Pool(nproc) as pool:
                    sols = pool.map(solver.run_simulation, list_J)
                logging.info(
                    "Computing errors and saving them in a two-columns file: dx and errors"
                )
                errs = np.empty((len(list_J) - 1, 2))
                np.savetxt(os.path.join(savedir, f"rho_J_{list_J[0]}"), sols[0][0])
                for i in range(1, len(list_J)):
                    np.savetxt(savedir + f"rho_J_{list_J[i]}", sols[i][0])
                    X1 = xmin + (np.arange(2 * list_J[i - 1] + 1) + 0.5) * L / (
                        2 * list_J[i - 1] + 1
                    )
                    X2 = xmin + (np.arange(2 * list_J[i] + 1) + 0.5) * L / (
                        2 * list_J[i] + 1
                    )
                    errs[i - 1, 0] = L / (2 * list_J[i - 1] + 1)
                    errs[i - 1, 1] = wasserstein(sols[i - 1][0], sols[i][0], X1, X2, p)
                logging.info("Finished computing errors %s", errs)
                np.savetxt(os.path.join(savedir, "errors.log"), errs, header="dx")

            elif cvg_param_name == "eps":
                J = 5000
                xmin, xmax = XMIN, XMAX
                L = xmax - xmin
                X = xmin + (np.arange(2 * J + 1) + 0.5) * L / (2 * J + 1)
                alpha = 1.0
                x0 = 0.5
                p = 2
                T = 0.5
                nbC = 0.9
                scheme = "upwind"
                savedir = os.path.join(
                    REPO_DIR,
                    TESTS_DIR,
                    "aggdiff_convergence_evolutif_"
                    + scheme
                    + f"_diffImplicite_wrt_eps_J_{J}_alpha_{alpha}_p_{p}_T_{T}_CFL_{nbC}/",
                )
                if os.path.exists(savedir):
                    run = input(
                        "The directory already exists. Please consider removing it before running the script. Do you want to run anyway? (y/n)"
                    )
                    if "n" in run:
                        exit()
                pathlib.Path(savedir).mkdir(parents=True, exist_ok=True)
                list_eps = [1 / 2**i for i in range(4, 25)]
                nproc = 4
                with Pool(nproc) as pool:
                    sols = pool.map(
                        lambda eps: Solver(
                            eps=eps, scheme=scheme, nbC=nbC
                        ).run_simulation(
                            J=J, xmin=xmin, xmax=xmax, alpha=alpha, x0=x0, T=T
                        ),
                        list_eps,
                    )
                logging.info(
                    "Computing errors and saving them in a two-columns file: dx and errors"
                )
                errs = np.empty((len(list_eps) - 1, 2))
                np.savetxt(os.path.join(savedir, f"rho_eps_{list_eps[0]}"), sols[0][0])
                for i in range(1, len(list_eps)):
                    np.savetxt(savedir + f"rho_eps_{list_eps[i]}", sols[i][0])
                    errs[i - 1, 0] = list_eps[i - 1]
                    errs[i - 1, 1] = wasserstein(sols[i - 1][0], sols[i][0], X, X, p)
                logging.info("Finished computing erros %s", errs)
                np.savetxt(os.path.join(savedir, "errors.log"), errs, header="eps")

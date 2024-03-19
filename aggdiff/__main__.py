import argparse
import logging
import os.path
import pathlib

import numpy as np
from multiprocess import Pool

from aggdiff.parameters import fgradW
from aggdiff.solver_evol import Solver
from aggdiff.utils import wasserstein

REPO_DIR = os.path.dirname(os.path.dirname((os.path.realpath(__file__))))
TESTS_DIR = "tests/"


def compute_solution(J: float, solver: Solver, T: float):
    return solver.compute_solution(J=J, T=T, plot=False)


def main() -> None:

    parser = argparse.ArgumentParser(
        prog="AggregationDiffusionSimulations",
        description="Running simulations and computing convergence errors for the aggregation-diffusion model with diffusion coefficient epsilon.",
    )
    parser.add_argument(
        "-simu",
        "--simulation",
        dest="run_simu",
        action="store_true",
        help="run simulation.",
    )
    parser.add_argument(
        "-plot",
        "--with-plot",
        dest="with_plot",
        action="store_true",
        help="if running a simulation, plot the solution of the simulation",
    )
    parser.add_argument(
        "-cvg",
        "--convergence-curves",
        dest="run_cvg",
        action="store_true",
        help="draw convergence curves.",
    )
    parser.add_argument(
        "--convergence-parameter",
        dest="cvg_param_name",
        type=str,
        choices={"eps", "dx"},
        help="convergence parameter when drawing convergence curves",
    )
    parser.add_argument("-eps", type=float, default=0.1, help="value for epsilon")
    parser.add_argument("-alpha", type=float, default=1.0, help="value for alpha")
    parser.add_argument("-x0", type=float, default=0.5, help="value for x0")
    parser.add_argument(
        "--xmin", type=float, default=-1.0, help="value for the left boundary"
    )
    parser.add_argument(
        "--xmax", type=float, default=1.0, help="value for the right boundary"
    )
    parser.add_argument(
        "--is-repulsive",
        dest="is_repulsive",
        action="store_true",
        help="apply to a is_repulsive potential (mutliplies the potential by -1)",
    )
    parser.add_argument(
        "--scheme",
        type=str,
        choices={"upwind", "theta-scheme", "rusanov", "lax-friedrichs"},
        default="theta-scheme",
        help="scheme to use",
    )
    parser.add_argument(
        "--theta",
        type=float,
        default=1.0,
        help="value for theta, the parameter for the theta-scheme",
    )
    parser.add_argument(
        "-nbC",
        type=float,
        default=0.9,
        help="CFL number",
    )
    parser.add_argument(
        "--c-rusanov",
        dest="c_rusanov",
        type=float,
        default=1.1,
        help="value for c in the Rusanov scheme",
    )
    parser.add_argument(
        "--boundary-conditions",
        dest="BC",
        type=str,
        choices={"periodic", "mass conservation"},
        default="mass conservation",
        help="Boundary conditions",
    )
    parser.add_argument(
        "-J",
        type=int,
        default=5000,
        help="value for J, the number of cells being 2*J+1",
    )
    parser.add_argument(
        "-T", type=float, default=1.0, help="value for T, the final time"
    )
    parser.add_argument("-p", type=float, default=2.0, help="value for p")
    parser.add_argument(
        "-v",
        "--verbose",
        dest="verbosity",
        action="count",
        default=0,
        help=(
            "Verbosity (between 1-4 occurrences with more leading to more "
            "verbose logging). ERROR=0, WARN=1, INFO=2, DEBUG=3"
        ),
    )
    args = parser.parse_args()

    log_level = {
        0: logging.ERROR,
        1: logging.WARN,
        2: logging.INFO,
        3: logging.DEBUG,
    }[min(args.verbosity, 3)]
    logging.captureWarnings(True)
    logging.basicConfig(
        filename="aggdiff.log",
        encoding="utf-8",
        level=log_level,
        format="%(asctime)s %(message)s",
        datefmt="%m/%d/%Y %I:%M:%S %p",
    )

    if args.run_simu:
        solver = Solver(
            eps=args.eps,
            fgradW=lambda x: fgradW(x, args.alpha, args.x0),
            is_repulsive=args.is_repulsive,
            xmin=args.xmin,
            xmax=args.xmax,
            scheme=args.scheme,
            theta=args.theta,
            nbC=args.nbC,
            c_rusanov=args.c_rusanov,
            BC=args.BC,
        )
        solver.compute_solution(J=args.J, T=args.T, plot=args.with_plot)

    if args.run_cvg:
        if args.cvg_param_name == "dx":
            # Warning: in the previous simulation, "scheme" was always set to "upwind"
            savedir = os.path.join(
                REPO_DIR,
                TESTS_DIR,
                "aggdiff_convergence_evolutif_"
                + args.scheme
                + f"_diffImplicite_wrt_dx_eps_{args.eps}_alpha_{args.alpha}_p_{args.p}_T_{args.T}_CFL_{args.nbC}/",
            )
            if os.path.exists(savedir):
                run = input(
                    "The directory already exists. Running the script will erase the previous simulations. Do you want to run anyway? (y/n)"
                )
                if "n" in run:
                    exit()
            pathlib.Path(savedir).mkdir(parents=True, exist_ok=True)
            list_J = [100 * 2**i for i in range(7)]
            list_args = [(J, args.T, False) for J in list_J]
            solver = Solver(
                eps=args.eps,
                fgradW=lambda x: fgradW(x, args.alpha, args.x0),
                is_repulsive=args.is_repulsive,
                xmin=args.xmin,
                xmax=args.xmax,
                scheme=args.scheme,
                theta=args.theta,
                nbC=args.nbC,
            )
            nproc = 2
            with Pool(nproc) as pool:
                sols = pool.starmap(solver.compute_solution, list_args)
            logging.info(
                "Computing errors and saving them in a two-columns file: dx and errors"
            )
            errs = np.empty((len(list_J) - 1, 2))
            np.savetxt(os.path.join(savedir, f"rho_J_{list_J[0]}"), sols[0][0])
            L = args.xmax - args.xmin
            for i in range(1, len(list_J)):
                np.savetxt(savedir + f"rho_J_{list_J[i]}", sols[i][0])
                X1 = args.xmin + (np.arange(2 * list_J[i - 1] + 1) + 0.5) * L / (
                    2 * list_J[i - 1] + 1
                )
                X2 = args.xmin + (np.arange(2 * list_J[i] + 1) + 0.5) * L / (
                    2 * list_J[i] + 1
                )
                errs[i - 1, 0] = L / (2 * list_J[i - 1] + 1)
                errs[i - 1, 1] = wasserstein(sols[i - 1][0], sols[i][0], X1, X2, args.p)
            logging.info("Finished computing errors %s", errs)
            np.savetxt(os.path.join(savedir, "errors.log"), errs, header="dx")

        elif args.cvg_param_name == "eps":
            L = args.xmax - args.xmin
            X = args.xmin + (np.arange(2 * args.J + 1) + 0.5) * L / (2 * args.J + 1)
            savedir = os.path.join(
                REPO_DIR,
                TESTS_DIR,
                "aggdiff_convergence_evolutif_"
                + args.scheme
                + f"_diffImplicite_wrt_eps_J_{args.J}_alpha_{args.alpha}_p_{args.p}_T_{args.T}_CFL_{args.nbC}/",
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
                        eps=eps,
                        fgradW=lambda x: fgradW(x, args.alpha, args.x0),
                        is_repulsive=args.is_repulsive,
                        xmin=args.xmin,
                        xmax=args.xmax,
                        scheme=args.scheme,
                        theta=args.theta,
                        nbC=args.nbC,
                    ).compute_solution(
                        J=args.J,
                        T=args.T,
                        plot=False,
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
                errs[i - 1, 1] = wasserstein(sols[i - 1][0], sols[i][0], X, X, args.p)
            logging.info("Finished computing erros %s", errs)
            np.savetxt(os.path.join(savedir, "errors.log"), errs, header="eps")


if __name__ == "__main__":
    main()

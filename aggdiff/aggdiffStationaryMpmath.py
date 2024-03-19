"""Code adapté du code de Benoît pour le papier aggdiff

Calcul des solutions stationnaires par une méthode de point fixe.

Tracé des solutions et ordre de convergence en eps à dx fixé."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import scipy as sp
import time
from multiprocessing import Pool
import pathlib
import ot
import os
import seaborn as sns
import mpmath
from mpmath import mp, mpf

#os.chdir('/homes/doua/trantien/documents/doctorat/codes/tests/')
os.chdir('/home/trantien/Documents/icj/doctorat/codes/tests/')

## Définir la précision

mpmath.mp.dps = 150 # 100 suffit sauf pour W(x)=sqrt(x)

## Décoration

sns.set_style('whitegrid')
palette = sns.color_palette('deep')

## Donnée initiale, paramètres
xmin, xmax = -1, 1
L = xmax - xmin

tol = 1e-15  # tolérance dans le schéma de point fixe

def rho0(x, J):
    xl = xmin + 0.25*L
    xr = xmin + 0.75*L
    return(np.exp(-20.*(x - (xmin + xmax)/2)**2))   # Une bosse
#    return(np.exp(-200.*(x - (xmin + xmax)/2)**2))   # Une bosse piquée
#    return(np.exp(-40.*(x - xl)**2) + np.exp(-40.*(x - xr)**2))    # Deux bosses
#    return(np.exp(-200.*(x + 0.2*L)**2) + np.exp(-200.*(x - 0.2*L)**2))    # Deux bosses loin et piquées
#    return(np.exp(-200.*(x + 0.3*L)**2) + np.exp(-200.*(x - 0.3*L)**2))    # Deux bosses et piquées et plus loin
#    return ((x - (xmin + xmax)/2 > -0.25)*(x - (xmin + xmax)/2 < 0.25)).astype(float)    # un 'Dirac rectangulaire'
#    return((x == (xmin+xmax)/2).astype(float))  # un seul Dirac
#    return((x == xmin + (int((2*J+1)/4) + 0.5) * L/(2*J + 1)).astype(float) + (x == xmin + (int(3*(2*J+1)/4) + 0.5) * L/(2*J + 1)).astype(float))     # deux Dirac

## Calculs de distances de Wasserstein

p = 1

def wasserstein(rho1, rho2, X1, X2, p):     # avec POT
    return ot.emd2_1d(X1, X2, rho1, rho2, metric='minkowski', p=p)**(1/p)
compute_error = lambda r1, r2, x1, x2: wasserstein(r1, r2, x1, x2, p)

def distToDirac(rho, X, p):     # la distance W_p au Dirac vaut (M_p(rho))^{1/p}
    return(np.sum(np.fabs(X)**p * rho) ** (1/p))    # ce truc reste en mpmath, incroyable

## Potentiels

# On n'a pas besoin de gradW

alpha = 0.
def w(x):
    '''Type 1'''    # Anguleux avec 2 puits en +/- x_0, C^1 sauf en 5 pts
    # p1, p2, p3, alpha1, alpha2 = 1, -0.9, 0.5, 0.2, 0.4     # 0 < alphai < 0.5
    # return((p1 * x) * (x < alpha1 * L) + (p2 * x +
    # (p1 - p2) * alpha1 * L) * (alpha1 * L <= x) * (x <= alpha2 * L)
    # + (p3 * x + (p2 - p3) * alpha2 * L +
    # (p1 - p2) * alpha1 * L) * (x > alpha2 * L))
    '''Type raté'''
    # a, b = (3*alpha1 - alpha2)*L/2, alpha1*L
    # cd = 1
    # return(cd*(x-a)*(x-b)**2 + cd*a*b**2)
    '''Type 2'''    # Pointu avec 2 puits en +/- x_0, C^1 sauf en 0
    # alpha1, alpha2 = 0.2, 0.4
    # a, alpha3 = -10, (0.5 + alpha2) / 2
    # b1 = - a * (alpha1 * L)**2
    # b2 = a * ((alpha1 - alpha2) * L)**2 / 2 + b1
    # return((a * (x - alpha1 * L)**2 + b1) * (x < (alpha1 + alpha2) * L / 2) +
    #  (b2 - a * (x - alpha2 * L)**2) * (x >= (alpha1 + alpha2) * L / 2) * (x <= alpha3 * L)
    #  + (a * (alpha2 - 0.5) * L * x + (b2 - a * ((alpha3 - alpha2) * L)**2) -
    #  a * (alpha2 - 0.5) * alpha3 * L**2) * (x > alpha3 * L))
    '''Type caucasien'''
    # return(x**(alpha+1))
    '''Puissance coupée linéairement'''
    x_0 = 0.5
    alpha = 1.
    return((x**(alpha+1)) * (x <= x_0) +
    ((alpha+1) * x_0**alpha * (x - x_0) + x_0**(alpha+1)) * (x > x_0))
    '''(A4-1) mais pas (A2)'''
    # return(np.sqrt(x) + x)

def W(x):   # W(x) = w(|x|)
    return(w(np.abs(x)))

# """Dessiner les potentiels en question"""
# J = 400 # 2*J + 1 points à l'intérieur
# dx = L/(2*J + 1)
# XW = np.arange(-(2*J), 2*J + 1)*dx
# fig = plt.figure()
# ax1 = fig.add_subplot(111)
# ax1.set_xlabel('x')
# ax1.grid(True)
# pp1, = ax1.plot(XW, W(XW))
# plt.show()

## Run

def run(eps, J, plot=False):

    dx = L/(2*J + 1)
    X = xmin + (np.arange(2*J + 1) + 0.5)*dx
    XW = np.arange(-(2*J), 2*J + 1)*dx
    rho = rho0(X, J)
    rhoIni = rho.copy()
    rho = list(rho)
    somme = mpmath.fsum(rho)
    for j in range(2*J+1):
        rho[j] = rho[j] / somme

    # rhoExacte = (1 - np.tanh(X / (2 * eps))**2) / (4 * eps) # Solution exacte pour W(x) = |x[
    # rhoExacte /= np.sum(rhoExacte)

    n = 0
    err = 10

    if plot:

        fig = plt.figure()
        fig.set_figheight(8)
        fig.set_figwidth(11)
        ax0 = plt.subplot(211)
        ax2 = plt.subplot(212)

        ax0.set_xlabel('x')
        ax0.set_ylabel(r'$\rho^{\varepsilon, 0}(x)$')
        pp0, = ax0.plot(X, rhoIni, linewidth=0.9, color = palette[0], linestyle = '--', label='Initial density')
        ax0.grid(False, axis = 'y')
        ax0.legend(loc='upper left')

        ax1 = ax0.twinx()
        ax1.set_ylabel(r'$\rho^{\varepsilon, n}(x)$')
        ax1.grid(True)
        pp1, = ax1.plot(X, rho, linewidth=0.9, color = palette[0], label = 'Solution')
        ax1.legend(loc='upper right')
        # pp11, = ax1.plot(X, rhoExacte, label = 'exacte', linestyle = '--') # Solution exacte pour W(x) = |x[

        ax2.set_xlabel('x')
        ax2.set_ylabel('W(x)')
        ax2.grid(True)
        pp2, = ax2.plot(XW, W(XW), linewidth=0.9, color = palette[3])

    while err > tol:

        # Schéma de point fixe
        rhoOld = rho.copy()
        convol = sp.signal.convolve(rho, W(XW), mode="same", method="fft")
        for j in range(2*J+1):
            rho[j] = mp.exp(-convol[j] / mpf(eps))
        somme = mpmath.fsum(rho)
        for j in range(2*J+1):
            rho[j] = rho[j] / somme

        if plot:

            pp1.set_ydata(rho)
            ax1.set_ylim([0, float(max(rho)) + 0.05*float(max(rho))])
            ax1.set_title(rf'Iteration n = {n} ; $\varepsilon$ = {eps}, J = {J}, tolerance = {tol}')
            plt.pause(0.2)

        # Actualisation de l'erreur
        err = compute_error(rhoOld, rho, X, X)     # erreur = distance entre deux solutions successives
        # err = distToDirac(rho, X, p)    # erreur = distance au Dirac en 0
        print(f'eps = {eps}')
        print(rf'W_p(rho^{n}, rho^{n+1}) = $', err)
        n += 1

    return(rho)
    #return(rho, n)   # Pour représentation d'ES mutiples

"""
Une simple simulation (run and plot)
"""
# eps = 0.001   # eps < 0.1, c'est bien
# J = 1000 # 2*J + 1 points à l'intérieur
#
# run(eps, J, True)
# plt.show()

"""
Etats stationnaires multiples
"""
# eps = 0.1   # eps < 0.1, c'est bien
# J = 5000 # 2*J + 1 points à l'intérieur
#
# def rho0(x, J):
#     xl = xmin + 0.25*L
#     xr = xmin + 0.75*L
#     return(np.exp(-20.*(x - (xmin + xmax)/2)**2))
#
# rho1, n1 = run(eps, J, False)
#
# def rho0(x, J):
#     xl = xmin + 0.25*L
#     xr = xmin + 0.75*L
#     return(np.exp(-200.*(x + 0.3*L)**2) + np.exp(-200.*(x - 0.3*L)**2))    # Deux bosses loin et piquées




"""
Convergence en eps avec dx fixé
"""
J = 5000

#savedir = f"aggdiff_convergence_stationnaire_upwind_diffImplicite_potentielMixte_eps_J_{J}_p_{p}_tol_{tol}/"
savedir = f"aggdiff_convergence_stationnaire_upwind_diffImplicite_alpha_{alpha}_lipschitz_eps_J_{J}_p_{p}_tol_{tol}/"
pathlib.Path(savedir).mkdir(parents=True, exist_ok=True)

eps = [1/2**i for i in range(4, 17)]
list_args = [(epsi, J) for epsi in eps]
nproc = 4
X = xmin + (np.arange(2*J + 1) + 0.5) * L / (2*J + 1)
XW = np.arange(-(2*J), 2*J + 1) * L / (2*J + 1)

with Pool(nproc) as pool:
    sols = pool.starmap(run, list_args)

# compute error norm
print("Computing errors")
errs = np.empty((len(list_args)-1, 2))
errs = []
#np.savetxt(savedir + f"rho_eps_{eps[0]}", sols[0])  # sauvegarde les rho, mais pas besoin pour ordre de cvg
for i in range(1, len(list_args)):
#    np.savetxt(savedir + f"rho_eps_{eps[i]}", sols[i]) # sauvegarde les rho, mais pas besoin pour ordre de cvg
    errs.append([eps[i-1], distToDirac(sols[i-1], X, p)])
#    errs[i-1, 0] = eps[i-1]
#    errs[i-1, 1] = compute_error(sols[i-1], sols[i], X, X)
#    errs[i-1, 1] = distToDirac(sols[i-1], X, p)

np.savetxt(savedir + "errors.log", errs, header="eps", fmt='%10.150f') # 100 suffit sauf pour W(x)=sqrt(x)

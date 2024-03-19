'''Pour tracer courbe de convergence à partir de fichiers d'erreurs.'''

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import mpmath
from mpmath import mp, mpf

#os.chdir('/homes/doua/trantien/documents/doctorat/codes/tests/')
os.chdir('/home/trantien/Documents/icj/doctorat/codes/tests/')

## Définir la précision

mpmath.mp.dps = 100

## Décoration

sns.set_style('whitegrid')
palette = sns.color_palette('deep')

## Evolutif : courbes de convergence sans mpmath

#toggle = 'Cvg en dx'
toggle = 'Cvg en eps'

file = 'aggdiff_convergence_evolutif_upwind_diffImplicite_eps_J_5000_alpha_1.0_lipschitz_p_2_T_0.5_CFL_0.9/errors.log'


if toggle == 'Cvg en dx':
    eps = float(file[file.find('eps') + 3 + 1:file.find('_', file.find('eps') + 3 + 1)])
else:
    J = float(file[file.find('J') + 1 + 1:file.find('_', file.find('J') + 1 + 1)])
p = int(file[file.find('p_') + 2:file.find('p_') + 3])
alpha = float(file[file.find('alpha') + 5 + 1:file.find('_', file.find('alpha') + 5 + 1)])
T = float(file[file.find('T') + 1 + 1:file.find('_', file.find('T') + 1 + 1)])
CFL = float(file[file.find('CFL') + 3 + 1:file.find('/')])

errs = np.loadtxt(file, skiprows=1)    # skiprows car row#1 = header
N = errs.shape[0]

# print("Estimation de l'ordre step by step :\n")
# for i in range(N-1):
#     order = (np.log(errs[i+1, 1]) - np.log(errs[i, 1])) / (np.log(errs[i+1, 0]) - np.log(errs[i, 0]))
#     print(f"{order}")
# print(r"Estimation de l'ordre sur tout le jeu de $\varepsilon$ :\n")
# order = 0
# for i in range(N-1):     # Il y a N-1 éléments...
#     order += (np.log(errs[i+1, 1]) - np.log(errs[i, 1])) / (np.log(errs[i+1, 0]) - np.log(errs[i, 0]))
# print(f"{order/(N-1)}")

plt.subplots(figsize=(11, 7))
plt.ylabel(rf'$\log_{{10}}$ of the error in $W_{{{p}}}$ distance')
plt.grid(True)

if toggle == 'Cvg en dx':
    plt.xlabel(r'-$\log_{10}(\Delta x)$')
    plt.title(rf'Convergence order in $W_{{{p}}}$ distance, upwind scheme with implicit diffusion, $\varepsilon = {eps:.3f}$, $W(x) = |x|^{{{alpha+1:.1f}}}$, $T = {T:.2f}$, $CFL = {CFL:.1f}$')
else:
    plt.xlabel(r'$\log_{10}(\varepsilon)$')
    plt.title(rf'Convergence order in $W_{{{p}}}$ distance, upwind scheme with implicit diffusion, $J = {J:.0f}$, $W(x) = |x|^{{{alpha+1:.1f}}}$, $T = {T:.2f}$, $CFL = {CFL:.1f}$')


# Tracé en log-log : on trace log(erreur)

start, stop = 0, N    #pour supprimer des points aberrants à la fin ou au début
start_k = start  # si on veut faire commencer la deuxième pente de référence plus tard

plt.plot(np.log10(errs[start:stop, 0]), np.log10(errs[start:stop, 1]), label="Scheme", marker='x', markersize = 7, color = palette[0])
# Pentes de référence
k = 0.5
plt.plot(np.log10(errs[start:start_k + 1, 0]), np.log10(errs[start_k + 1, 1]) + k * (np.log10(errs[start:start_k + 1, 0]) - np.log10(errs[start_k + 1, 0])), linestyle ='--', color = palette[1])
plt.plot(np.log10(errs[start_k:stop, 0]), np.log10(errs[start_k, 1]) + k * (np.log10(errs[start_k:stop, 0]) - np.log10(errs[start_k, 0])), label=f"Slope {k}", color = palette[1])
plt.plot(np.log10(errs[start:stop, 0]), np.log10(errs[start, 1]) + np.log10(errs[start:stop, 0]) - np.log10(errs[start, 0]), label="Slope 1", color = palette[2])

plt.legend()
plt.savefig('aggdiff_convergence_evolutif_upwind_diffImplicite_eps_J_5000_alpha_1.0_lipschitz_p_2_T_0.5_CFL_0.9.png', dpi=300)
plt.show()

## Stationnaire : courbes de convergence avec mpmath

# toggle = 'Cvg en eps'
#
# file = 'aggdiff_convergence_stationnaire_upwind_diffImplicite_potentielMixte_eps_J_50000_p_1_tol_1e-15/errors.log'
#
# J = float(file[file.find('J') + 1 + 1:file.find('_', file.find('J') + 1 + 1)])
# p = int(file[file.find('p_') + 2:file.find('p_') + 3])
# #alpha = float(file[file.find('alpha') + 5 + 1:file.find('_', file.find('alpha') + 5 + 1)])
# tol = float(file[file.find('tol') + 3 + 1:file.find('/')])
#
# errs = np.loadtxt(file, skiprows=1, dtype = mpmath.ctx_mp_python.mpf)    # skiprows car row#1 = header
# N = len(errs)
# start, stop = 0, N    #pour supprimer des points aberrants à la fin ou au début
#
# # print("Estimation de l'ordre step by step :\n")
# # for i in range(N-1):
# #     order = (mpmath.log(errs[i+1, 1]) - mpmath.log(errs[i, 1])) / (mpmath.log(errs[i+1, 0]) - mpmath.log(errs[i, 0]))
# #     print(f"{order}")
# print(r"Estimation de l'ordre sur tout le jeu de $\varepsilon$ :\n")
# order = 0
# for i in range(start, stop-1):     # Il y a N-1 éléments...
#     order += (mpmath.log(errs[i+1, 1]) - mpmath.log(errs[i, 1])) / (mpmath.log(errs[i+1, 0]) - mpmath.log(errs[i, 0]))
# print(f"{float(order/(stop-start-1)):.8f}")
#
# plt.subplots(figsize=(11, 7))
# plt.ylabel(rf'$\log_{{10}}$ of the error in $W_{{{p}}}$ distance')
# plt.grid(True)
#
# plt.xlabel(r'$\log_{10}(\varepsilon)$')
# plt.title(rf'Convergence order in $W_{{{p}}}$ distance, $J = {J:.0f}$, $W(x) = |x|^{{{alpha+1:.1f}}}$, tolerance = {tol}')
#
# # Tracé en log-log : on trace log(erreur)
# logEps, logErrors = [mpmath.log10(errs[i, 0]) for i in range(stop)], [mpmath.log10(errs[i, 1]) for i in range(stop)]
# plt.plot(logEps, logErrors, label="Fixed-point method", marker='x', linewidth = .8, color = palette[0])
# # Pentes de référence
# k = 2
# plt.plot(logEps, [logErrors[start] + k * (logEps[i] - logEps[start]) for i in range(stop)], label=f"Slope {k}", linewidth = .8, color = palette[1])
# #plt.plot(logEps, [logErrors[start] + logEps[i] - logEps[start] for i in range(stop)], label="Slope 1", linewidth = .8, color = palette[2])
#
# plt.legend()
# #plt.savefig('aggdiff_convergence_stationnaire_upwind_diffImplicite_potentielMixte_eps_J_80000_p_1_tol_1e-12.png', dpi=300)
# plt.show()

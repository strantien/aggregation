'''Script de visualisation

A partir de agreg.py'''

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

## Déco

sns.set_style('whitegrid')
palette = sns.color_palette('deep')

## Lecture données

#os.chdir('/homes/doua/trantien/documents/doctorat/codes/tests/')
os.chdir('/home/trantien/Documents/icj/doctorat/codes/tests/')
file1 = 'schema_upwind_eps_0.0_b_0_T_0.6_L_2_J_200_alpha_2_M_1_CFL_0.9_c_1.1'
file2 = 'schema_upwind_eps_0.0_b_0_T_1_L_2_J_200_alpha_2_M_1_CFL_0.9_c_1.1'

schema1 = file1[file1.find('schema') + 6 + 1:file1.find('_', file1.find('schema') + 6 + 1)]
schema2 = file2[file2.find('schema') + 6 + 1:file2.find('_', file2.find('schema') + 6 + 1)]
eps = file1[file1.find('eps') + 3 + 1:file1.find('_', file1.find('eps') + 3 + 1)]
T1 = float(file1[file1.find('T') + 1 + 1:file1.find('_', file1.find('T') + 1 + 1)])
T2 = float(file2[file2.find('T') + 1 + 1:file2.find('_', file1.find('T') + 1 + 1)])

X = np.loadtxt(file1 + '_X')
rhoIni = np.loadtxt(file1 + '_rhoIni')
aIni = np.loadtxt(file1 + '_aIni')
a1 = np.loadtxt(file1 + '_a')
a2 = np.loadtxt(file2 + '_a')
rho1 = np.loadtxt(file1 + '_rho')
rho2 = np.loadtxt(file2 + '_rho')

## Visualisation : rhoIni, rho avec 1 ou 2 schémas, a

# fig = plt.figure()
# ax1 = plt.subplot(311)
# ax2 = plt.subplot(312)
# ax3 = plt.subplot(313)
# fig.set_figheight(8)
# fig.set_figwidth(11)
#
# ax1.set_title(r'Donnée initiale, $T = 0$')
# rhoIniMin, rhoIniMax = np.amin(rhoIni), np.amax(rhoIni)
# ax1.set_xlim([-0.5, 0.5])       # A la main
# ax1.set_ylim([rhoIniMin - 0.05*(rhoIniMax - rhoIniMin), rhoIniMax + 0.05*(rhoIniMax - rhoIniMin)])
# ax1.plot(X, rhoIni, linewidth=0.9, color = palette[4], linestyle = '-')
# ax1.set_xlabel(r'$x$')
# ax1.set_ylabel(r'$\rho_{\Delta x}(t=0)$')
# ax1.grid(True)
#
# ax2.set_title(rf'Densité à $T = {T1:.2f}$')
# ax2.set_xlim([-0.5, 0.5])
# #ax2.set_ylim([rhoIniMin - 0.05*(rhoIniMax - rhoIniMin), rhoIniMax + 0.05*(rhoIniMax - rhoIniMin)]) # axes fixes
# ax2.set_ylim([np.amin(rho1) - 0.05*(np.amax(rho1) - np.amin(rho1)), np.amax(rho1) + 0.05*(np.amax(rho1) - np.amin(rho1))])
# ax2.plot(X, rho1, linewidth=0.9, color = palette[0], linestyle = '-', label = schema1.capitalize())
# ax2.plot(X, rho2, linewidth=0.9, color = palette[2], linestyle = '--', label = schema2.capitalize())
# ax2.set_xlabel(r'$x$')
# ax2.set_ylabel(rf'$\rho_{{\Delta x}}(t={T1:.2f})$')
# ax2.legend()
# ax2.grid(True)
#
# ax3.set_title(rf'Vitesse à $T = {T1:.2f}$')
# ax3.set_xlim([-0.5, 0.5])       # A la main
# #ax3.set_ylim([-0.05, 0.05])       # A la main
# ax3.plot(X, a1, linewidth=0.9, color = palette[1])
# ax3.set_xlabel(r'$x$')
# ax3.set_ylabel(rf'$a_{{\Delta x}}(t={T1:.2f})$')
# ax3.grid(True)

## Visualisation : rhoIni, aIni, rho(T1), a(T1), rho(T2), a(T2) avec 1 seul schéma

fig = plt.figure()
ax1 = plt.subplot(321)
ax2 = plt.subplot(322)
ax3 = plt.subplot(323)
ax4 = plt.subplot(324)
ax5 = plt.subplot(325)
ax6 = plt.subplot(326)
fig.set_figheight(8)
fig.set_figwidth(11)

ax1.set_title(r'Donnée initiale, $T = 0$')
rhoIniMin, rhoIniMax = np.amin(rhoIni), np.amax(rhoIni)
#ax1.set_xlim([-0.5, 0.5])       # A la main
ax1.set_ylim([rhoIniMin - 0.05*(rhoIniMax - rhoIniMin), rhoIniMax + 0.05*(rhoIniMax - rhoIniMin)])
ax1.plot(X, rhoIni, linewidth=0.9, color = palette[4], linestyle = '-')
ax1.set_xlabel(r'$x$')
ax1.set_ylabel(r'$\rho_{\Delta x}(t=0)$')
ax1.grid(True)

ax2.set_title(r'Vitesse initiale, $T = 0$')
#ax2.set_xlim([-0.5, 0.5])       # A la main
#ax2.set_ylim([-0.05, 0.05])       # A la main
ax2.plot(X, aIni, linewidth=0.9, color = palette[1])
ax2.set_xlabel(r'$x$')
ax2.set_ylabel(r'$a_{\Delta x}(t=0)$')
ax2.grid(True)

ax3.set_title(rf'Densité à $T = {T1:.2f}$')
ax3.set_xlim([-0.5, 0.5])
#ax3.set_ylim([rhoIniMin - 0.05*(rhoIniMax - rhoIniMin), rhoIniMax + 0.05*(rhoIniMax - rhoIniMin)]) # axes fixes
ax3.set_ylim([np.amin(rho1) - 0.05*(np.amax(rho1) - np.amin(rho1)), np.amax(rho1) + 0.05*(np.amax(rho1) - np.amin(rho1))])
ax3.plot(X, rho1, linewidth=0.9, color = palette[4], linestyle = '-')
ax3.set_xlabel(r'$x$')
ax3.set_ylabel(rf'$\rho_{{\Delta x}}(t={T1:.2f})$')
ax3.grid(True)

ax4.set_title(rf'Vitesse à $T = {T1:.2f}$')
ax4.set_xlim([-0.5, 0.5])       # A la main
#ax4.set_ylim([-0.05, 0.05])       # A la main
ax4.plot(X, a1, linewidth=0.9, color = palette[1])
ax4.set_xlabel(r'$x$')
ax4.set_ylabel(rf'$a_{{\Delta x}}(t={T1:.2f})$')
ax4.grid(True)

ax5.set_title(rf'Densité à $T = {T2:.2f}$')
ax5.set_xlim([-0.05, 0.05])
#ax5.set_ylim([rhoIniMin - 0.05*(rhoIniMax - rhoIniMin), rhoIniMax + 0.05*(rhoIniMax - rhoIniMin)]) # axes fixes
ax5.set_ylim([np.amin(rho2) - 0.05*(np.amax(rho2) - np.amin(rho2)), np.amax(rho2) + 0.05*(np.amax(rho2) - np.amin(rho1))])
ax5.plot(X, rho2, linewidth=0.9, color = palette[4], linestyle = '-')
ax5.set_xlabel(r'$x$')
ax5.set_ylabel(rf'$\rho_{{\Delta x}}(t={T2:.2f})$')
ax5.grid(True)

ax6.set_title(rf'Vitesse à $T = {T2:.2f}$')
ax6.set_xlim([-0.05, 0.05])       # A la main
#ax6.set_ylim([-0.05, 0.05])       # A la main
ax6.plot(X, a2, linewidth=0.9, color = palette[1])
ax6.set_xlabel(r'$x$')
ax6.set_ylabel(rf'$a_{{\Delta x}}(t={T2:.2f})$')
ax6.grid(True)


## Affichage et sauvegarde

fig.tight_layout()
plt.savefig('tralala', dpi=300)
plt.show()
"""Code general avec ou sans diffusion"""
"""Diffusion classique ou fractionnaire"""
"""Agrégation, optimisation : Benoît Fabrèges"""
"""LapFrac : code de Maxime Herda"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import scipy as sp
import scipy.signal
from scipy.sparse import diags
import scipy.sparse.linalg as scspl
import argparse
import time

# Donnée initiale et paramètres
xmin, xmax = -1, 1
L = xmax - xmin
J = 200 # 2*J + 1 points à l'intérieur
dx = L/(2*J + 1)
X = xmin + (np.arange(2*J + 1) + 0.5)*dx

chi = 1.
eps = 0. # Coefficient de diffusion
alph = 2 # alpha dans (0,2]

Mcrit = 2 * np.pi / chi # Masse critique pour diffusion classique
M = 1 # Masse totale initiale (conservée si schéma conservatif...)

nbC = 0.9 # Nombre CFL
c = 1.1 # Constante c dans Rusanov

T = 2   # Temps final
t = 0
dt = 1

def rho0(x):
    xl = xmin + 0.25*L
    xr = xmin + 0.75*L
#    return(np.exp(-20.*(x - (xmin + xmax)/2)**2))   # Une bosse
    return(np.exp(-40.*(x - xl)**2) + np.exp(-40.*(x - xr)**2))    # Deux bosses
#    return ((x - (xmin + xmax)/2 > -0.25)*(x - (xmin + xmax)/2 < 0.25)).astype(float)    # un 'Dirac rectangulaire'
#    return((x == (xmin+xmax)/2).astype(float))  # un seul Dirac

rho = rho0(X)
rho = M*rho/np.sum(rho)
rhoIni = rho.copy()
a = np.zeros(2*J + 1)


if alph == 2: # Laplacien classique en format sparse
    d = np.ones(2*J + 1)
    s = np.ones(2*J)
    I = diags([d],[0],format="csr")
    Lap = diags([s,-2*d,s],[-1,0,1],format="csr")
    Lap[0,0], Lap[2*J, 2*J] = -1, -1 # CL  de conservation de la masse
    Lap /= dx**2

else: # Laplacien fractionnaire en format classique
      # Attention, on calcule les coeffs de -Lap, et je fais Lap <- - Lap à la fin
    evenratio = 2
    K = evenratio * J + 1
    LW = K * dx     # Domaine de calcul de la convolution : [-LW, LW]
    Calph = alph*2**(alph-1)*sp.special.gamma((alph+1)/2)/(np.sqrt(np.pi)*sp.special.gamma((2-alph)/2))

    beta = np.zeros(K)
    if alph != 1:
        phi = lambda t: t**(2-alph)/(2-alph)/(alph-1)/alph
        dphi = lambda t: t**(1-alph)/(alph-1)/alph
        ddphi = lambda t: -t**(-alph)/alph
    else:
        phi = lambda t: t-t*np.log(t)
        dphi = lambda t: -np.log(t)
        ddphi = lambda t: -1/t

    #calcul des beta_k, k=1...K
    #attention, beta[k] vaut beta_{k+1} car k commence à 0 en Python
    beta[0] = - ddphi(1) - (dphi(3)+3*dphi(1))/2 + phi(3) - phi(1) + 1/(2-alph)
    k = np.arange(1,K+1)
    kev = k[k%2 == 0]
    kod = k[k%2 == 1][1:]   #il faut exclure k=1, traité auparavant
    beta[kev-1] = 2*(dphi(kev+1)+dphi(kev-1)-phi(kev+1)+phi(kev-1))
    beta[kod-1] = -(dphi(kod+2)+6*dphi(kod)+dphi(kod-2))/2+phi(kod+2)-phi(kod-2)
    beta[-1] = -(dphi(K-2)+3*dphi(K))/2-phi(K-2)+phi(K)+ddphi(K)

    #calcul final des beta_k pour k=-K...K
    #maintenant, beta[k] vaut beta_{k-K}
    beta = Calph * beta/dx**(alph+1)
    beta = np.concatenate((np.flip(beta),[0],beta))

    Lap = np.zeros((2*J+1,2*J+1))

    for i in range(2*J+1): # i goes from 0 to 2*J

        Lap[i,i] = np.sum(beta) * dx +  2 * Calph / (alph * LW**alph) # optimisable en sortant la somme des beta ?

        # Convolution in [-L,L]
        for j in range(2*J+1):

            Lap[i,j] = Lap[i,j] - beta[i-j+K] * dx

    Lap = - Lap

# Gradient du potentiel
XgradW = np.arange(-(2*J), 2*J + 1)*dx

b = 1
if b == 0:  # W(x) ~ x^2
    gradW = XgradW
if b == 1:   # W(x) = |x|
    gradW = np.sign(XgradW)
if b == 2:  # W(x) = (chi/pi)*ln(|x|)
    gradW = (chi/(np.pi*XgradW))
else:
    gradW = XgradW/(np.abs(XgradW))**b # b = 3/2 => W ~ sqrt

gradW[2*J] = 0      # Convention a_chapo

def compute_velocity(rho, kernel):
    return(-sp.signal.convolve(rho, kernel, mode="same", method="fft"))

def compute_time_step(rho, a, dx, dt, nbC, t, T):

    vel_pos = a > 0
    vel_neg = a <= 0

    arho = a*rho

    # CFL condition : pas la meme en fonction du schema
    dt = nbC*dx/np.amax(np.fabs(a)) # Upwind
#    dt = nbC*dx**2/np.amax(np.fabs(a)) # Parabolique
#    dt = nbC*dx/c # Rusanov/LF

    dt = min(dt, T-t)

    rhs = np.zeros_like(rho)

    # Upwind
    rhs[1:-1] -= (dt/dx)*np.fabs(a[1:-1])*rho[1:-1]
    rhs[:-1] -= (dt/dx)*arho[1:]*vel_neg[1:]
    rhs[1:] += (dt/dx)*arho[:-1]*vel_pos[:-1]
    if eps > 0: # CL  de conservation de la masse bis
         rhs[0] -= (dt/dx)*a[0]*rho[0]
         rhs[-1] += (dt/dx)*a[-1]*rho[-1]

    # Le bord des schémas suivants n'a pas été adapté pour utilisation avec eps > 0

    # Rusanov
    # rhs[1:-1] -= 0.5 * (arho[2:] - arho[:-2])
    # rhs[1:-1] += 0.5 * c * (rho[2:] - 2 * rho[1:-1] + rho[:-2])

    # Carrillo/Lax-Friedrichs
    # rhs[1:-1] = 0.25 * (-(a[1:-1] + a[2:]) * (rho[1:-1] + rho[2:]) + (a[:-2] + a[1:-1]) * (rho[:-2] + rho[1:-1]))
    # rhs[1:-1] += 0.5 * c * (rho[2:] - 2 * rho[1:-1] + rho[:-2])

    # Minmod Solem (schema a 5 points)
    # Dans ce schema, le a global n'est pas utilise
    # rhoRightPlus, rhoRightMinus, rhoLeftPlus, rhoLeftMinus = np.zeros_like(rho), np.zeros_like(rho), np.zeros_like(rho), np.zeros_like(rho)
    # rhoRightPlus[2:-2] = rho[3:-1] - 0.5 * (np.minimum(rho[3:-1], rho[4:]) - np.minimum(rho[2:-2], rho[3:-1]))
    # rhoRightMinus[2:-2] = rho[2:-2] + 0.5 * (np.minimum(rho[2:-2], rho[3:-1]) - np.minimum(rho[1:-3], rho[2:-2]))
    # rhoLeftPlus[2:-2] = rho[2:-2] - 0.5 * (np.minimum(rho[2:-2], rho[3:-1]) - np.minimum(rho[1:-3], rho[2:-2]))
    # rhoLeftMinus[2:-2] = rho[1:-3] + 0.5 * (np.minimum(rho[1:-3], rho[2:-2]) - np.minimum(rho[:-4], rho[1:-3]))
    # JRight = 0.5 * (compute_velocity(rhoRightPlus, gradW) * rhoRightPlus + compute_velocity(rhoRightMinus, gradW) * rhoRightMinus) + (c / 2) * (rhoRightPlus - rhoRightMinus)
    # JLeft = 0.5 * (compute_velocity(rhoLeftPlus, gradW) * rhoLeftPlus + compute_velocity(rhoLeftMinus, gradW) * rhoLeftMinus) + (c / 2) * (rhoLeftPlus - rhoLeftMinus)
    # rhs = JRight - JLeft

    rho += rhs

    if eps > 0:

        if alph == 2:
            A = I - eps*dt*Lap
            rho[:] = scspl.spsolve(A, rho)     # Diffusion classique, calcul en sparse
        else:
            rho[:] = sp.linalg.solve(np.eye(2*J + 1) - eps*dt*Lap, rho)   # Diffusion classique, calcul normal

    return(dt)

# Temps de fonctionnement du script
tIni = time.time()

## Trace au temps T

# fig = plt.figure()
# ax1 = plt.subplot(211)
# ax2 = plt.subplot(212)
#
# ax1.set_title(f"T = 0")
# rhoIniMin, rhoIniMax = np.amin(rhoIni), np.amax(rhoIni)
# ax1.set_ylim([rhoIniMin - 0.05*(rhoIniMax - rhoIniMin), rhoIniMax + 0.05*(rhoIniMax - rhoIniMin)])
# ax1.plot(X, rho, linewidth=0.9)
# ax1.grid()
#
# # Trace au temps T
#
# while t < T:
#
#     a = compute_velocity(rho, gradW)
#     dt = compute_time_step(rho, a, dx, dt, nbC, t, T)
#     t += dt
#
# ax2.set_title(f"T = {T:.3f}")
# #ax2.set_ylim([rhoIniMin - 0.05*(rhoIniMax - rhoIniMin), rhoIniMax + 0.05*(rhoIniMax - rhoIniMin)]) # axes fixes
# ax2.set_ylim([np.amin(rho) - 0.05*(np.amax(rho) - np.amin(rho)), np.amax(rho) + 0.05*(np.amax(rho) - np.amin(rho))])
# ax2.plot(X, rho, linewidth=0.9)
# ax2.grid()
#
# fig.tight_layout()
# plt.show()

## Animation

def gen():
    global t, T
    it = 0
    while t < T:
        it += 1
        yield it

def update(frame_number):
    global dt, t

    a = compute_velocity(rho, gradW)
    dt = compute_time_step(rho, a, dx, dt, nbC, t, T)
    t += dt
#    rhoIniMin, rhoIniMax = np.amin(rhoIni), np.amax(rhoIni)
#    aIni = compute_velocity(rhoIni, gradW)
#    aIniMin, aIniMax = np.amin(aIni), np.amax(aIni)
#    ax1.set_ylim([rhoIniMin - 0.05*(rhoIniMax - rhoIniMin), rhoIniMax + 0.05*(rhoIniMax - rhoIniMin)]) # axes fixes
    ax1.set_ylim([np.amin(rho) - 0.05*(np.amax(rho) - np.amin(rho)), np.amax(rho) + 0.05*(np.amax(rho) - np.amin(rho))])
    ax1.set_title("t = " + f"{t:.3f}")
    ax1.set_ylabel(r'$\rho$')
    pp1.set_ydata(rho)
    ax2.set_ylim([np.amin(a) - 0.05*(np.amax(a) - np.amin(a)), np.amax(a) + 0.05*(np.amax(a) - np.amin(a))])
#    ax2.set_ylim([aIniMin - 0.05*(aIniMax - rhoIniMin), aIniMax + 0.05*(aIniMax - aIniMin)]) # axes fixes
    ax2.set_ylabel(r'$a[\rho]$')
    pp2.set_ydata(a)
    # print('dt =', dt)
    print('t =', t)
    print('Masse totale =', np.sum(rho))
    return(pp1, pp2)



# print the initial solution
fig = plt.figure()
fig.canvas.manager.full_screen_toggle()
ax1 = plt.subplot(211)
ax1.set_title(f"{t:.3f}")
ax1.grid()
pp1, = ax1.plot(X, rho)
#pp1, = ax1.plot(X, rho, '.-')
ax2 = plt.subplot(212)
pp2, = ax2.plot(X, compute_velocity(rho, gradW))
#pp2, = ax2.plot(X, compute_velocity(rho, gradW), '.-')
ax2.grid()

anim = animation.FuncAnimation(fig, update, frames=gen, interval=30, repeat=False)
plt.show()

print('Le script a tourné pendant', time.time()-tIni, 'secondes.')
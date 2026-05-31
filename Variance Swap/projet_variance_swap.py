import numpy as np
import matplotlib.pyplot as plt

# Box-Muller : génère deux gaussiennes indépendantes depuis des uniformes
def bx_mu(n,N):
    U,V = np.random.rand(n,N), np.random.rand(n,N)
    return np.sqrt(-2*np.log(U))*np.cos(2*np.pi*V), np.sqrt(-2*np.log(U))*np.sin(2*np.pi*V)

# CDF de la loi normale centrée réduite (Abramowitz & Stegun)
def Abramowitz(x):
    if x<0:
        return 1-Abramowitz(-x)
    else:
        b0 = 0.2316419
        b1 = 0.319381530
        b2 = -0.356563782
        b3 = 1.781477937
        b4 = -1.821255978
        b5 = 1.330274429
        t = 1/(1+b0*x)
        return 1 - np.exp(-x**2 /2) * (b1*t+b2* t**2 +b3* t**3 +b4* t**4 +b5 * t**5)/np.sqrt(2*np.pi)


# Q1 : St = S0 * exp(intégrale sur [0,t] de (r-0.5*sig(s)**2)ds + sig(s)dWs)

# Q2 : on utilise le même type de démonsatrtion que pour les options asiatiques, on split l'intrégrale de [t0,T] en [t0,0] et en [0,T],
# en posant S_hist l'intégrale sur [t0,0] et S_barre la moyenne standart de St sur [0,T],
# on obtient le payoff suivant : h = T*S_barre/(T-t0) - K_eff**2 avec K_eff**2 = -S_hist/(T-t0) + K**2

# Q3 : si on utilise le modèles de Black & Sholes donc avec sigma constant, on obtient un payoff qui n'est aps une VA,
# donc VS_theo = exp(-r*T) * (sig**2 - K**2) qui est déterministe. Si le strike est équitable, on aurait une payoff nul


# Q4 :  tout comme avec Black & Sholes, on pose Yt = log(sig(t)) et on applique Itô à Yt,
# on obtient, dlog(sig(t)) = dsig(t)/sig(t) - 0.5 * (dsig(t)/sig(t))**2, on utilise l'EDS, et en passant à l'exponentielle,
# on obtient : sig(t) = sig_0 * exp(nu*Zt - 0.5 * nu**2 * t)
# on utilise Fubini et on sait que sig suit une loi log-normale, on en déduit donc que espérance de intégrale sur [0,T] de sig(s)**2 ds = 
# sig(0)**2 * (exp(nu**2 *T)-1)/nu**2, d'où : VS_théo = exp(-r*T) * (sig(0)**2 * (exp(nu**2 *T)-1)/nu**2 - K**2)


# Q5 : 

# Paramètres du modèle à volatilité stochastique
nu = 0.8
S0 = 1
sig_0 = 0.35
rho = -0.5
K = 0.35
r = 0.01
T=3/12
N=63  # nb de pas quotidiens sur 3 mois (≈ T/dt avec dt=1/252)

q_95 = 1.645  # quantile 90% pour l'IC

# Q5 : simulation des incréments corrélés dW et dZ via décomposition de Cholesky
# dZ = rho*dW + sqrt(1-rho²)*eps pour obtenir Corr(dW,dZ)=rho
def mbs(T,N,rho,eps1,eps2):
    dt = T/N
    dw = np.sqrt(dt)*eps1
    dz = np.sqrt(dt)*(rho*eps1 + np.sqrt(1-rho**2)*eps2)
    return dw,dz

# Cumsum pour reconstituer W(t) et Z(t) depuis leurs incréments
def traject_mbs(T,N,rho,eps1,eps2):

    dw,dz = mbs(T,N,rho,eps1,eps2)

    W = np.cumsum(dw,axis=1)
    Z = np.cumsum(dz,axis=1)

    # Ajout de W(0)=Z(0)=0 en tête de trajectoire
    W = np.concatenate([np.zeros((W.shape[0],1)), W], axis=1)
    Z = np.concatenate([np.zeros((Z.shape[0],1)), Z], axis=1)    

    return W,Z

# Q6 : on connaît dejà la formule fermé de sigma, cependant pour St, on va utiliser une discrétisation d'Euler :

def traject(T,N,rho,nu,S0,sig_0,eps1,eps2):
    dt = T/N

    W,Z = traject_mbs(T,N,rho,eps1,eps2)
    S= np.zeros_like(W)
    t = np.linspace(0,T,N+1)

    # sigma(t) = sig_0 * exp(nu*Z(t) - 0.5*nu²*t) : formule fermée (Q4)
    sig = sig_0 * np.exp(nu*Z - 0.5 * nu**2 * t)
    S[:,0] = S0
    dW = W[:,1:]-W[:,:-1]
    # Schéma d'Euler log-normal : S(t+dt) = S(t) * exp((r - 0.5*sig²)*dt + sig*dW)
    for i in range(N):
        S[:,i+1] = S[:,i] * np.exp((r-0.5*sig[:,i]**2)*dt + sig[:,i]*dW[:,i])
    return S,sig

n=50000           #nombre de simulation

# Estimateur MC standard de VS_real : variance réalisée discrète = (1/T)*sum(log(S_i/S_{i-1})²)
def VS_MC(T,N,rho,nu,K,S0,sig_0,n):

    eps1,eps2 = bx_mu(n,N)

    S = traject(T,N,rho,nu,S0,sig_0,eps1,eps2)[0]
    log_S = np.log(S[:,1:]/S[:,:-1])
    X = np.sum(log_S**2,axis=1)/T  # variance réalisée par trajectoire
    VS = np.exp(-r*T)*(np.mean(X)-K**2)

    ecart_t = np.std(X,ddof=1)

    IC = np.array([VS-q_95*ecart_t/np.sqrt(n),VS+q_95*ecart_t/np.sqrt(n)])

    return VS,IC



# Q7 : variables antithétiques : on moyenne les payoffs de (eps1,eps2) et (-eps1,-eps2)
# réduit la variance si X(eps) et X(-eps) sont négativement corrélés

def VS_MC_anti(T,N,rho,nu,K,S0,sig_0,n):

    eps1,eps2 = bx_mu(n,N)

    S_plus = traject(T,N,rho,nu,S0,sig_0,eps1,eps2)[0]
    S_moins = traject(T,N,rho,nu,S0,sig_0,-eps1,-eps2)[0]
    log_S_plus = np.log(S_plus[:,1:]/S_plus[:,:-1])
    log_S_moins = np.log(S_moins[:,1:]/S_moins[:,:-1])
    # Estimateur antithétique : moyenne des deux variances réalisées
    X = 0.5*(np.sum(log_S_plus**2,axis=1)+np.sum(log_S_moins**2,axis=1))/T
    VS = np.exp(-r*T)*(np.mean(X)-K**2)

    ecart_t = np.std(X,ddof=1)

    IC = np.array([VS-q_95*ecart_t/np.sqrt(n),VS+q_95*ecart_t/np.sqrt(n)])

    return VS,IC


# Grille logarithmique de n pour observer la convergence
n_list = np.unique(np.logspace(3, 6, 9, dtype=int))
n_taille = np.size(n_list)

VS_MC_list = np.zeros(n_taille)
VS_MC_anti_list = np.zeros(n_taille)

IC_list = np.zeros((n_taille,2))
IC_anti_list = np.zeros((n_taille,2))


for i,n in enumerate(n_list):
    VS_MC_list[i],IC_list[i,:] = VS_MC(T,N,rho,nu,K,S0,sig_0,n)
    VS_MC_anti_list[i],IC_anti_list[i,:] = VS_MC_anti(T,N,rho,nu,K,S0,sig_0,n)

# VS_theo = exp(-rT) * (sig0² * (exp(nu²T)-1)/nu² - K²)  (Q4)
VS_theo = 0.010318432



plt.figure()
plt.title("estimateurs de VS en fonction de n")
plt.plot(n_list,VS_MC_list,label="Monte Carlo")
plt.plot(n_list,VS_MC_anti_list,label="Monte Carlo anithétique")
plt.fill_between(n_list,IC_list[:,0],IC_list[:,1],alpha=0.3,label="IC 90%")
plt.fill_between(n_list,IC_anti_list[:,0],IC_anti_list[:,1],alpha=0.3,label="IC anti 90%")
plt.axhline(y=VS_theo,color="black")
plt.xscale("log")
plt.yscale("log")
plt.xlabel("nombre de simulations")
plt.ylabel("prix de l'option")
plt.legend()
plt.show()

plt.figure()
plt.title("différence des estimateurs par rapport à la théorie en fonction de n")
plt.plot(n_list,np.abs(VS_theo-VS_MC_list)/np.abs(VS_theo),label="Monte Carlo")
plt.plot(n_list,np.abs(VS_theo-VS_MC_anti_list)/np.abs(VS_theo),label="Monte Carlo antithétique")
plt.xscale("log")
plt.yscale("log")
plt.xlabel("nombre de simulations")
plt.ylabel("différence en fonction de la valeur théorique")
plt.legend()
plt.show()



# Q8 : pour n=50000, on a une erreur inférieur à 5%, on va donc utiliser n=50000 pour ces questions
# VS_real croît avec nu : plus nu est grand, plus sigma est volatile → variance réalisée plus élevée
# L'écart VS_real - VS_theo augmente aussi avec nu (approximation discrète moins précise)

n=50000

nbr_nu = 40
nu_list = np.linspace(0,1.5,nbr_nu)
VS_nu_list = np.zeros(nbr_nu)
VS_nu_anti_list = np.zeros(nbr_nu)

IC_nu_list = np.zeros((nbr_nu,2))
IC_nu_anti_list = np.zeros((nbr_nu,2))

for i,nu in enumerate(nu_list):
    VS_nu_list[i],IC_nu_list[i,:] = VS_MC(T,N,rho,nu,K,S0,sig_0,n)
    VS_nu_anti_list[i],IC_nu_anti_list[i,:] = VS_MC_anti(T,N,rho,nu,K,S0,sig_0,n)

plt.figure()
plt.title("Estimateurs de VS en fonction de nu")
plt.plot(nu_list,VS_nu_list,label="Monte Carlo")
plt.plot(nu_list,VS_nu_anti_list,label="Monte Carlo antithétique")
plt.fill_between(nu_list,IC_nu_list[:,0],IC_nu_list[:,1],alpha=0.3,label="IC 90%")
plt.fill_between(nu_list,IC_nu_anti_list[:,0],IC_nu_anti_list[:,1],alpha=0.3,label="IC anti 90%")
plt.xlabel("nu")
plt.ylabel("prix de l'option")
plt.legend()
plt.show()

plt.figure()
plt.title("Diff des estimateurs en fonction de nu")
plt.plot(nu_list,np.abs(VS_nu_list-VS_nu_anti_list),color="black",label="diff des estimateurs")
plt.xlabel("nu")
plt.ylabel("différence estimateurs")
plt.legend()
plt.show()


# Q9 : VS_real et VS_theo sont indépendants de rho (la variance intégrée ne dépend pas de la corrélation W-Z)
# La différence VS_real - VS_theo reste donc quasi-constante en rho

n=50000
nu = 0.8


nbr_rho = 30
rho_list = np.linspace(-0.99,0.99,nbr_rho)
VS_nu_list = np.zeros(nbr_rho)
VS_nu_anti_list = np.zeros(nbr_rho)

IC_nu_list = np.zeros((nbr_rho,2))
IC_nu_anti_list = np.zeros((nbr_rho,2))

for i,rho in enumerate(rho_list):
    VS_nu_list[i],IC_nu_list[i,:] = VS_MC(T,N,rho,nu,K,S0,sig_0,n)
    VS_nu_anti_list[i],IC_nu_anti_list[i,:] = VS_MC_anti(T,N,rho,nu,K,S0,sig_0,n)

plt.figure()
plt.title("Estimateurs de VS en fonction de rho")
plt.plot(rho_list,VS_nu_list,label="Monte Carlo")
plt.plot(rho_list,VS_nu_anti_list,label="Monte Carlo antithétique")
plt.fill_between(rho_list,IC_nu_list[:,0],IC_nu_list[:,1],alpha=0.3,label="IC 90%")
plt.fill_between(rho_list,IC_nu_anti_list[:,0],IC_nu_anti_list[:,1],alpha=0.3,label="IC anti 90%")
plt.xlabel("rho")
plt.ylabel("prix de l'option")
plt.legend()
plt.show()

plt.figure()
plt.title("Diff des estimateurs en fonction de rho")
plt.plot(rho_list,np.abs(VS_nu_list-VS_nu_anti_list),color="black",label="diff des estimateurs")
plt.xlabel("rho")
plt.ylabel("différence estimateurs")
plt.legend()
plt.show()



# Q10 : utillisation de Taylor reste intégrale et décomposition si k<x ou k>x


# Q11 : utilisation de l'EDS pour St et en posant Xt = ln(St). D'après Itô, on obtient : 
# dSt/St = dln(St) + 0.5 * sig(t)**2 dt, d'où sig(t)**2 = 2dSt/St - 2dln(St). On intègre sur [0,T], on a:
# intégrale sur [0,T] de sig(t)**2 dt = 2*intégrale sur [0,T] de dSt/St - 2*(ln(St)-ln(S0)).
# En posant, ensuite f(x) = ln(x) et en utilisant la question 10 et avec réarrangement des termes, on obtient la réponse.


# Q12 : la variance swap peut être répliqué statiquement par un portefeuille continue de puts et de calls OTM pondérés en 1/k**2.
# Celà provient de la représentation du log-contract via l'identité de breeden-litzenberg appliqué à f(x) = ln(x)


# Q15 : densité de S(T) pour différentes valeurs de nu
# nu=0 → lognormal pur (B&S), nu>0 → queues plus épaisses (vol de vol élevée)

nu_list = np.array([0,0.75,1.5])
n=500000
rho=-0.5

plt.figure()
plt.title("densité de ST pour différentes valeurs de nu")
for nu in nu_list:
    eps1,eps2 = bx_mu(n,N)
    St = traject(T,N,rho,nu,S0,sig_0,eps1,eps2)[0][:,-1]
    plt.hist(St[St<3],bins=100,density=True,alpha=0.3,label=f"nu={nu}")
plt.legend()
plt.xlabel("x")
plt.ylabel("densité de ST")
plt.show()


# Q16 : densité de S(T) pour différentes valeurs de rho
# rho<0 → skew négatif (queues gauches plus épaisses), rho>0 → skew positif

rho_list = np.array([-0.75,0,0.75])
n=500000
nu=0.8

plt.figure()
plt.title("densité de ST pour différentes valeurs de rho")
for rho in rho_list:
    eps1,eps2 = bx_mu(n,N)
    St = traject(T,N,rho,nu,S0,sig_0,eps1,eps2)[0][:,-1]
    plt.hist(St[St<3],bins=100,density=True,alpha=0.3,label=f"rho={rho}")
plt.legend()
plt.xlabel("x")
plt.ylabel("densité de ST")
plt.show()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.interpolate import CubicSpline
from scipy.integrate import solve_ivp
#constants
G=6.67430e-8
M_solar_sun=1.989e33
R_earth=6.378e8
c=3e10
hbar=1.05e-27
m_e=9.10938356e-28
hbar=1.054571817e-27
m_u=1.66053906660e-24
mu_e=2.0

D_th=(m_u*(m_e**3)*(c**3)*(mu_e**3))/(3.0*np.pi**2*(hbar**3))
C_th=m_e**4*c**5/(24*np.pi**2*hbar**3)

#----part b---
def read_wd_data(filename):
    #reads the file as logg and M
    data=pd.read_csv(filename)
    mass=data["mass"]
    logg=data["logg"]
    g=10**logg
    mass_sun=M_solar_sun*mass.values
    R=np.sqrt(G*mass_sun/g)
    R_earth_units=R/R_earth
    plt.figure(figsize=(6,5))
    plt.scatter(mass, R_earth_units, s=20, alpha=0.7)
    plt.xlabel(r"Mass in Solar Mass[$M$]")
    plt.ylabel(r"Radius R (in Earth units)")
    plt.title("White Dwarf Mass–Radius Relation")
    plt.grid(True)
    plt.show()
    return mass,R_earth_units
#-----part c----
def low_value(mass, R):
    #finds the fit for the low M and returns the relation between R and M
    mass_limited = np.array([m for m in mass if m < 0.4])
    radius_limited = np.array([R[i] for i, m in enumerate(mass) if m < 0.4])
    log_m=np.log(mass_limited).reshape(-1, 1)
    log_r=np.log(radius_limited)
    reg=LinearRegression()
    reg.fit(log_m,log_r)
    coefs=reg.coef_[0]
    interc=np.exp(reg.intercept_)
    M_fit=np.linspace(mass_limited.min(), mass_limited.max(), 200)
    R_fit=interc*M_fit**coefs
    plt.scatter(mass_limited, radius_limited, s=20, alpha=0.7, label="Low-mass data")
    plt.loglog(M_fit, R_fit, 'r', lw=2, label=f"Power-law fit R = {interc:.4f} * M^{coefs:.4f}")
    plt.xlabel(r"Mass in Solar Mass[$M$]")
    plt.ylabel(r"Radius R (in Earth units)")
    plt.title("White Dwarf Mass–Radius Relation (Low Mass)")
    plt.grid(True,which="both")
    plt.legend()
    plt.show()
    return interc, coefs

def n_q(coefs):
    #return n and q 
    n=(1-3*coefs)/(1-coefs)
    q=5*n/(n+1)
    print(f"n={n}, q={q}")
    return n,q
#----part d---
def C_eq(K,D,q):
    C=5*K*D**(5/q)/(8*q)
    return C
#I got help from GPT for the four functions below
def equations(r,y,D,C):
    m,rho=y
    if rho<=0:
        return [0, 0]
    dmdr=4*np.pi*r**2*rho
    P_rho = dPdrho(rho, D, C)
    drhodr=-G*m*rho/(r**2*P_rho)
    return [dmdr,drhodr]

def dPdrho(rho, D, C):
    x = (rho / D)**(2/3)
    gamma = (5 + 2*x) / (3 * (1 + x)) 
    P = C * (rho/D)**(5/3) / np.sqrt(1 + x)
    return gamma * (P / rho)

def surface_event(r, y, D, C):
    return y[1] 
surface_event.terminal = True
surface_event.direction = -1

def interpolate(mass,radius):
    spline=CubicSpline(radius,mass)
    return spline 
def sample(rho_c_min,rho_c_max,N):
    #takes N number of samples for rho_c
    rho_c_values=np.logspace(np.log10(rho_c_min),np.log10(rho_c_max),N)
    return rho_c_values
def error(mass_real,mass_interp):
    err=np.sqrt(np.sum((mass_real-mass_interp)**2))
    return err
def ivp_solver(rho_c,D,C):
    #solves ivp for functions in equations given initial values for r,m,rho
    r0=1e-4
    r_final=1e10
    m0=4/3*np.pi*r0**3*rho_c
    y0=[m0,rho_c]
    solution=solve_ivp(equations,(r0,r_final),y0,args=(D,C),events=surface_event)
    if len(solution.t_events[0]) == 0:
        return np.nan, np.nan
    R=solution.t_events[0][0]
    M=solution.y_events[0][0][0]
    M_vals=M/M_solar_sun   
    R_vals=R/R_earth   
    return R_vals,M_vals
        
def part_d(M,R,D,N,q,K):
    #iterates oves rho_c to find D value and calculates error
    rho_c_min=1e-3*D
    rho_c_max=D
    rho_c_values=sample(rho_c_min,rho_c_max,N)
    R_vals=[]
    M_vals=[]
    C=C_eq(K,D,q)
    for rho in rho_c_values:
        Radius,Mass=ivp_solver(rho,D,C)
        R_vals.append(Radius)
        M_vals.append(Mass)
    R_vals=np.array(R_vals)
    M_vals=np.array(M_vals)
    mask = np.isfinite(R_vals) & np.isfinite(M_vals)
    R_vals = R_vals[mask]
    M_vals = M_vals[mask]
    index=np.argsort(R_vals)
    R_vals=R_vals[index]
    M_vals=M_vals[index]
    if len(R_vals) < 2:
        return np.inf
    spline=interpolate(M_vals,R_vals)
    mass_spline=spline(R)
    err=error(M,mass_spline)
    return err

def minimize_D(M,R,N,q,K):
    #iterates over D_grid to find the less error
    err=[]
    D_grid=np.logspace(5,7,40)
    for D in D_grid:
        err.append(part_d(M,R,D,N,q,K))
    err=np.array(err)
    i=np.argmin(err)
    D=D_grid[i]
    err_min=err[i]
    print("Best D",D)
    plt.loglog(D_grid,err)
    plt.title("Error vs D")
    plt.xlabel("D")
    plt.ylabel("Error")
    plt.grid(True)
    plt.show()
    return D

def part_e(D, q, K, N=80):
    #to fit D values and find max M
    rho_c_values = np.logspace(-3, 1, N) * D
    C = C_eq(K, D, q)
    R_vals = []
    M_vals = []
    for rho_c in rho_c_values:
        R, M = ivp_solver(rho_c, D, C)
        if np.isfinite(R) and np.isfinite(M):
            R_vals.append(R)
            M_vals.append(M)
    R_vals = np.array(R_vals)
    M_vals = np.array(M_vals)
    idx = np.argsort(R_vals)
    R_vals = R_vals[idx]
    M_vals = M_vals[idx]
    return R_vals, M_vals


def part_eplot(D, q, K):
    #just for plotting of e 
    R_vals, M_vals = part_e(D, q, K)
    M_Ch_num = np.max(M_vals)
    R_at_MCh = R_vals[np.argmax(M_vals)]
    print(f"Numerical Chandrasekhar mass = {M_Ch_num:.3f} M_sun")
    M_Ch_theory = 1.44
    print(f"Theoretical Chandrasekhar mass = {M_Ch_theory:.3f} M_sun")
    plt.figure(figsize=(7,5))
    plt.plot(R_vals, M_vals, lw=2, label="Numerical M–R curve")
    plt.axhline(M_Ch_theory, color='r', ls='--', label="Chandrasekhar mass (theory)")
    plt.xlabel(r"Radius $R$ (Earth radii)")
    plt.ylabel(r"Mass $M$ ($M solar mass$)")
    plt.title("White Dwarf Mass–Radius Relation")
    plt.legend()
    plt.grid(True)
    plt.show()
    return M_Ch_num, R_at_MCh


def newton():
    file="/Users/ilknur/final_project_414/white_dwarf_data.csv"
    mass,R=read_wd_data(file)
    interc,coefs=low_value(mass,R)
    n,q=n_q(coefs)
    K=2792194809105.1196  
    q=3
    D=minimize_D(mass,R,40,q,K)
    print("Dtheory=",D_th)
    print("Ctheory=",C_th)
    C=C_eq(K,D,q)
    print("Best C=",C)

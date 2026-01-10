import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

G=6.67430e-8
M_solar_sun=1.989e33
R_earth=6.378e8

def read_wd_data(filename):
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

def low_value(mass, R):
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
    plt.xlabel(r"Mass [$M_\odot$]")
    plt.ylabel(r"Radius [$R_\oplus$]")
    plt.title("White Dwarf Mass–Radius Relation (Low Mass)")
    plt.grid(True,which="both")
    plt.legend()
    plt.show()
    return interc, coefs

def n_q(coefs):
    n=(1-3*coefs)/(1-coefs)
    q=5*n/(n+1)
    print(f"n={n}, q={q}")
    return n,q

file="/Users/ilknur/final_project_414/white_dwarf_data.csv"
mass,R=read_wd_data(file)
interc,coefs=low_value(mass,R)
n_q(coefs)
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


G=6.67430e-11 #m**3*kg-1*s-2
M_solar_sun=1.989e30 #kg
R_earth=6.378e6 #m 
rho_r0=1e16 #kg/m**3
c=3e8 #m/s
L_unit=G*M_solar_sun/(c**2)
#rho0=c**2/(G*L_unit**2)
p_c2=1.5689e31*0.1/(c**2)
p0=c**2/(G*L_unit**2)

def get_K(gamma):
    K_si=p_c2/(rho_r0**gamma)
    K=(K_si*(p0**gamma)/p0)
    return K
def p_to_rho(p_geo, K,gamma):
    if p_geo <= 0:
        return 0, 0
    rho_r = (p_geo/K)**(1/gamma) #geo
    rho_geo = rho_r + K/(gamma - 1)*rho_r**gamma #geo
    return rho_geo, rho_r #geo,geo

def TOV(r,y,K,gamma):
    m,p_geo,v=y
    rho_geo,rho_r=p_to_rho(p_geo,K,gamma)
    denom=r*(r-2*m)
    if denom<=0:
        return [0,0,0]
    dm=4*np.pi*r**2*rho_geo
    dv=2*(m+np.pi*4*r**3*p_geo)/denom
    dp=-(m+np.pi*4*r**3*p_geo)/denom*(p_geo+rho_geo)
    return [dm,dp,dv]
def TOV_b(r,y,K,gamma):
    m,p_geo,v,mp=y
    rho_geo,rho_r=p_to_rho(p_geo,K,gamma)
    denom=r*(r-2*m)
    if denom<=0 or p_geo<=0:
        return [0,0,0,0]
    dm=4*np.pi*r**2*rho_geo
    dv=2*(m+np.pi*4*r**3*p_geo)/denom
    dp=-(m+np.pi*4*r**3*p_geo)/denom*(p_geo+rho_geo)
    dmP=4*np.pi*r**2*rho_r/np.sqrt(1-2*m/r)
    return [dm, dp, dv, dmP]
#got help from GPT for this function
def surface_event(r, y):
    return y[1] 
surface_event.terminal = True
surface_event.direction = -1

def integrate(p_geo,r_max,K,gamma): #using the same structure from newton part d
    r0=0
    v0=0
    m0=0
    y0=[m0,p_geo,v0]
    sol=solve_ivp(lambda r,y:TOV(r,y,K,gamma),(r0,r_max),y0,events=surface_event,rtol=1e-6, atol=1e-10,
    max_step=0.05)
    if sol.status == 1:
        R_geo = sol.t[-1]
        M_geo = sol.y[0, -1]
        return R_geo, M_geo
    return np.nan,np.nan
def integrate_b(p_geo,r_max,K,gamma):
    r0=0
    v0=0
    m0=0
    y0=[m0,p_geo,v0,0]
    sol=solve_ivp(lambda r, y: TOV_b(r, y, K,gamma),(r0,r_max),y0,events=surface_event,rtol=1e-6, atol=1e-10,
    max_step=0.05)
    if sol.status==1:
        M=sol.y[0, -1]   
        MP=sol.y[3, -1] 
        R =sol.t[-1]
        Delta=(MP-M)/M
        return M,MP,R,Delta
    return np.nan,np.nan,np.nan,np.nan
def part_a(gamma):
    #return M vs R graph 
    K=get_K(gamma)
    p_grid=np.logspace(-12,-2,50)#geo p
    M_vals=[]
    R_vals=[]
    r_max=50
    for p in p_grid:
        R,M=integrate(p,r_max,K,gamma)
        if np.isfinite(R) and np.isfinite(M):
            R_vals.append(R*L_unit/1000) #km
            M_vals.append(M) #molar mass
    plt.figure(figsize=(6, 5))
    plt.plot(R_vals, M_vals, marker="o")
    plt.xlabel("Radius R (km)")
    plt.ylabel("Mass M (M_sun)")
    plt.title(f"Neutron Star M-R Curve for {gamma}")
    plt.grid(True)
    plt.show()

def part_b(gamma):
    #return R vs Delta graph
    K=get_K(gamma)
    p_grid=np.logspace(-12,-2,40)#geo p
    M_vals=[]
    R_vals=[]
    D=[]
    MP=[]
    r_max=100
    for p in p_grid:
        M,mp,R,Delta=integrate_b(p,r_max,K,gamma)
        if np.isfinite(R) and np.isfinite(M):
            R_vals.append(R*L_unit/1000) #km
            M_vals.append(M) #molar mass
            D.append(Delta)
            MP.append(mp)
    plt.figure(figsize=(6, 5))
    plt.plot(R_vals, D, marker="o")
    plt.xlabel("Radius R (km)")
    plt.ylabel("Delta")
    plt.title(f"Neutron Star Delta-R Curve for {gamma}")
    plt.grid(True)
    plt.show()
    plt.plot(MP,M_vals, marker="o")
    plt.xlabel("M")
    plt.ylabel("MP")
    plt.title(f"Neutron Star M-MP Curve for {gamma}")
    plt.grid(True)
    plt.show()


def part_c(gamma):
    #returns plots for M vs pc and Mp vs pc
    K=get_K(gamma)
    p_grid=np.logspace(-12,-1,40)#geo p
    M_vals=[]
    R_vals=[]
    D=[]
    MP=[]
    rho_c_list=[]
    r_max=100
    for p in p_grid:
        M,mp,R,Delta=integrate_b(p,r_max,K,gamma)
        if np.isfinite(R) and np.isfinite(M):
            rho_c_list.append((p/K)**(1/gamma))
            R_vals.append(R*L_unit/1000) #km
            M_vals.append(M) #molar mass
            D.append(Delta)
            MP.append(mp)
    rho_c=np.array(rho_c_list)
    M=np.array(M_vals)
    dMdp=np.gradient(M, rho_c)
    stable=dMdp > 0
    unstable=dMdp < 0
    plt.loglog(rho_c[stable], M[stable], 'o-', label='Stable')
    plt.loglog(rho_c[unstable], M[unstable], 'o--', label='Unstable')
    imax = np.argmax(M)
    plt.loglog(rho_c[imax], M[imax], 'ro', label=r'$M_{\max}$')
    plt.xlabel(r'$p_c$')
    plt.ylabel(r'$M$')
    plt.title(f"M vs pc for {gamma}")
    plt.legend()
    plt.grid()
    plt.show()
    pc=np.array(rho_c_list)
    Mp=np.array(MP)
    dMdp=np.gradient(Mp, pc)
    stable=dMdp>0
    unstable=dMdp<0
    plt.loglog(rho_c[stable], Mp[stable], 'o-', label='Stable')
    plt.loglog(rho_c[unstable], Mp[unstable], 'o--', label='Unstable')
    imax = np.argmax(M)
    plt.loglog(rho_c[imax], M[imax], 'ro', label=r'$M_{\max}$')
    plt.xlabel(r'$p_c$')
    plt.ylabel(r'$M_p$')
    plt.title(f"Mp vs pc for {gamma}")
    plt.legend()
    plt.grid()
    plt.show()
    print(f"rho_c_max{rho_c[imax]}, M_max={M[imax]}")
    print(f"rho_c_max{rho_c[imax]}, Mp_max={Mp[imax]}")

def part_d():
    gamma_list=[1.3569,2.7138]
    K_values=np.logspace(-1, 1, 10)
    r_max=100
    for gamma in gamma_list:
        Mmax_list=[]
        for K_test in K_values:
            p_grid=np.logspace(-12, -2, 40)
            M_vals=[]
            for p in p_grid:
                R, M=integrate(p, r_max, K_test,gamma)
                if np.isfinite(M):
                    M_vals.append(M)
            if len(M_vals)>10:
                Mmax_list.append(max(M_vals))
            else:
                Mmax_list.append(np.nan)
        Mmax = np.array(Mmax_list)
        plt.loglog(K_values,Mmax,label=f"{gamma}",marker="o")
    plt.title("K vs Mmax")
    line=[2.5 for i in K_values]
    plt.loglog(K_values,line)
    plt.grid(True,which="both")
    plt.xlabel("K")
    plt.ylabel("M_max(K)")
    plt.legend()
    plt.show()


def einstein():
    gamma_list=[1.3569,2.7138]
    for gamma in gamma_list:
        part_a(gamma)
        part_b(gamma)
        part_c(gamma)
    part_d()

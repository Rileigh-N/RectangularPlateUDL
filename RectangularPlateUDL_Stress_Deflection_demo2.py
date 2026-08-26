import numpy as np
import sympy as sp

#initialization: increasing M,N yields higher accuracy at the cost of much slower calculation speed
x, y = sp.symbols('x y', real=True)
M, N = 4, 4

#support conditions
#0 = free, 1 = simply supported, 2 = fixed
#when q UDL across entire plate and fixed on opposite edges and free on opposite edges - analyzed using Euler-Bernoulli beam theory
a0=2
a1=2
b0=0
b1=0

#plate geometry 
#l = a1 - a0 // d = b1-b0
h=8 #in
l=96 #in
d=96 #in

#material properties
Y=29000000 #Elastic Modulus (psi)
v=0.2 #Poisson's Ratio (in/in)
q=x #Uniformly Distributed Load (psi)
D=Y*h**3/(12*(1-v**2)) #flexural ridigity (lb-in)

#equations governing deflection based on supports 
def X (i,x,l):
  return (x/l)**(a0)*(1 - x/l)**(a1)*(x/l)**(i)
def Y (j,y,d):
   return (y/d)**(b0)*(1 - y/d)**(b1)*(y/d)**(j)

c = sp.symbols('c1:'+str(M*N+1))  #Rayleigh's Coefficients

#creating array of the sums of X and Y from 1 to M,N
phi_list = []
idx = 0
for i in range(1, M+1):
    for j in range(1, N+1):
        phi_ij = X(i, x, l)*Y(j, y, d)
        phi_list.append(phi_ij)
        idx += 1

#Ritz Series (summing prior array)
w = sum(c[k]*phi_list[k] for k in range(M*N))

#derivatives taken symbolically
w_xx = sp.diff(w, x, 2)
w_yy = sp.diff(w, y, 2)
w_xy = sp.diff(w, x, y)

#total strain energy 
U = (D/2)*sp.integrate((w_xx + w_yy)**2 - 2*(1 - v)*(w_xx*w_yy - w_xy**2), (x, 0, l), (y, 0, d))

#potential of load 
V = -sp.integrate(q*w, (x, 0, l), (y, 0, d))

TotalEnergy = sp.simplify(U + V)

#equations for Kc=F matrix
eqns = [sp.diff(TotalEnergy, ci) for ci in c]

#solve matrix for Rayleigh coefficients
sol = sp.solve(eqns, c, dict=True)[0]

#plug found coefficients into symbolic deflection function w(x,y)
w_sol = w.subs(sol)
#make symbolic function into numerical / able to be evaluate numerically
w_num = sp.lambdify((x, y), w_sol, 'numpy')

#moments and stresses
#redefining derivatives to reduce computation time (symbollically)
w_xx_sol = sp.diff(w_sol, x, 2)
w_yy_sol = sp.diff(w_sol, y, 2)
w_xy_sol = sp.diff(w_sol, x, y)

Mx_sol  = -D*(w_xx_sol + v*w_yy_sol)
My_sol  = -D*(w_yy_sol + v*w_xx_sol)
Mxy_sol = -D*(1 - v)*w_xy_sol

#enable above to be evaluated numerically
Mx_num  = sp.lambdify((x, y), Mx_sol, 'numpy')
My_num  = sp.lambdify((x, y), My_sol, 'numpy')
Mxy_num = sp.lambdify((x, y), Mxy_sol, 'numpy')

StressX_num = lambda X, Y: 6*Mx_num(X, Y)/h**2
StressY_num = lambda X, Y: 6*My_num(X, Y)/h**2
StressT_num = lambda X, Y: 6*Mxy_num(X, Y)/h**2

#von Mises stress
VonMises_num = lambda X, Y: np.sqrt(StressX_num(X, Y)**2 - StressX_num(X, Y)*StressY_num(X, Y) + StressY_num(X, Y)**2)

# grid search for max deflection and max von Mises
nx, ny = 64, 64
xs = np.linspace(0, l, nx)
ys = np.linspace(0, d, ny)
Xg, Yg = np.meshgrid(xs, ys)

wg  = w_num(Xg, Yg)
vmg = VonMises_num(Xg, Yg)

max_defl = wg.max()
idx_defl = np.unravel_index(np.argmax(wg), wg.shape)
x_defl, y_defl = Xg[idx_defl], Yg[idx_defl]

max_vm = vmg.max()
idx_vm = np.unravel_index(np.argmax(vmg), vmg.shape)
x_vm, y_vm = Xg[idx_vm], Yg[idx_vm]

print("Max deflection (in):", float(max_defl))
print("Location (x,y) (in):", float(x_defl), float(y_defl))

print("Max von Mises stress (psi):", float(max_vm))
print("Location (x,y) (in):", float(x_vm), float(y_vm))
from pylab import *
from casadi import *

T = 1.0 # control horizon [s]
N = 400 # Number of control intervals

dt = T/N # length of 1 control interval [s]

##
# ----------------------------------
#    continuous system dot(x)=f(x,u)
# ----------------------------------

# Construct a CasADi function for the ODE right-hand side

A = np.array(
    [[0,0,0,0,0,0,1,0],
     [0,0,0,0,0,1,0,1],
     [0,0,0,0,1,0,0,1],
     [0,0,0,1,0,0,0,1],
     [0,0,1,1,1,1,1,1],
     [0,1,0,0,0,0,0,1],
     [1,0,0,0,0,0,0,1],
     [1,1,1,0,0,0,1,1]])
nx = A.shape[0]
B = np.array(
 [[-0.073463,-0.073463],
  [-0.146834,-0.146834],
  [-0.146834,-0.146834],
  [-0.146834,-0.146834],
  [-0.446652,-0.446652],
  [-0.147491,-0.147491],
  [-0.147491,-0.147491],
  [-0.371676,-0.371676]])

nu = B.shape[1]

x  = MX.sym('x',nx)
u  = MX.sym('u',nu)

dx = sparsify(A) @ sqrt(x)+sparsify(B) @ u

x_steady = (-solve(A,B @ vertcat(1,1)))**2

# Continuous system dynamics as a CasADi Function
f = Function('f', [x, u], [dx])

# -----------------------------------------------
#    Optimal control problem, multiple shooting
# -----------------------------------------------

opti = casadi.Opti()

# Decision variables for states
X = opti.variable(nx,N+1)
# Decision variables for control vector
U = opti.variable(nu,N)

xn = MX.sym("xn",x.numel())
xc = MX.sym("xc",x.numel(),4)

G = Function('G',[xn,xc, x,u],[
	vertcat(
	xc[:,0]-f(x, u),
	xc[:,1]-f(x + dt/2 * xc[:,0], u),
	xc[:,2]-f(x + dt/2 * xc[:,2], u),
	xc[:,3]-f(x + dt * xc[:,3], u),
	xn-(x+dt/6*(xc[:,0] +2*xc[:,1] +2*xc[:,2] + xc[:,3]))
	)],{"cse":True})
G = G.expand()

Gmap = G.map(N)

K = opti.variable(nx,4*N)

# Gap-closing shooting constraints
opti.subject_to(Gmap(X[:,1:],K,X[:,:-1],U)==0)

# Path constraints
opti.subject_to(0.01 <= (vec(X) <= 0.1))

# Initial guesses
opti.set_initial(X, repmat(x_steady,1,N+1))
opti.set_initial(U, 1)

# Initial and terminal constraints
opti.subject_to(X[:,0]==x_steady)
# Objective: regularization of controls

xbar = opti.variable()
opti.minimize(1e-6*sumsqr(U)+sumsqr(X[:,-1]-xbar))

# solve optimization problem
options = {}
#options["jit"] = True
#options["jit_options"] = {"flags":["-O3","--fast-math","-march=native"]}
options["madnlp"] = {"linear_solver":"cudss"}
opti.solver('madnlp',options)

sol = opti.solve()

figure()
spy(sol.value(jacobian(opti.g,opti.x)))

##
# -----------------------------------------------
#    Post-processing: plotting
# -----------------------------------------------

figure()
Xsol = sol.value(X)
plot(Xsol.T,'o-')

show()

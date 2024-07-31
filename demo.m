close all
clear all
import casadi.*

T = 1; % control horizon [s]
N = 40; % Number of control intervals

dt = T/N; % length of 1 control interval [s]

%%
% ----------------------------------
%    continuous system dot(x)=f(x,u)
% ----------------------------------

% Construct a CasADi function for the ODE right-hand side

A = [0 0 0 0 0 0 1 0;
     0 0 0 0 0 1 0 1;
     0 0 0 0 1 0 0 1;
     0 0 0 1 0 0 0 1;
     0 0 1 1 1 1 1 1;
     0 1 0 0 0 0 0 1;
     1 0 0 0 0 0 0 1;
     1 1 1 0 0 0 1 1];
nx = size(A,1);
B = [  -0.073463  -0.073463;
  -0.146834  -0.146834;
  -0.146834  -0.146834;
  -0.146834  -0.146834;
  -0.446652  -0.446652;
  -0.147491  -0.147491;
  -0.147491  -0.147491;
  -0.371676  -0.371676];

nu = size(B,2);

x  = MX.sym('x',nx);
u  = MX.sym('u',nu);

dx = sparse(A)*sqrt(x)+sparse(B)*u;

x_steady = (-A\B*[1;1]).^2;

% Continuous system dynamics as a CasADi Function
f = Function('f', {x, u}, {dx});

% -----------------------------------------------
%    Optimal control problem, multiple shooting
% -----------------------------------------------

opti = casadi.Opti();

% Decision variables for states
X = opti.variable(nx,N+1);
% Decision variables for control vector
U = opti.variable(nu,N);

xn = MX.sym('xn',x.numel());
xc = MX.sym('xc',x.numel(),4);

G = Function('G',{xn,xc, x,u},{
	[
	xc(:,1)-f(x, u);
	xc(:,2)-f(x + dt/2 * xc(:,1), u);
	xc(:,3)-f(x + dt/2 * xc(:,3), u);
	xc(:,4)-f(x + dt * xc(:,4), u);
	xn-(x+dt/6*(xc(:,1) +2*xc(:,2) +2*xc(:,3) + xc(:,4)))
	]},struct('cse',true));
G = G.expand();

Gmap = G.map(N);

K = opti.variable(nx,4*N);

% Gap-closing shooting constraints
opti.subject_to(Gmap(X(:,2:end),K,X(:,1:end-1),U)==0);

% Path constraints
opti.subject_to(0.01 <= vec(X) <= 0.1);

% Initial guesses
opti.set_initial(X, repmat(x_steady,1,N+1));
opti.set_initial(U, 1);

% Initial and terminal constraints
opti.subject_to(X(:,1)==x_steady);
% Objective: regularization of controls

xbar = opti.variable();
opti.minimize(1e-6*sumsqr(U)+sumsqr(X(:,end)-xbar));

% solve optimization problem
options = struct();
options.madnlp.linear_solver = 'cudss';
opti.solver('madnlp',options);

sol = opti.solve();

figure()
spy(sol.value(jacobian(opti.g,opti.x)))

%%
% -----------------------------------------------
%    Post-processing: plotting
% -----------------------------------------------

figure()
Xsol = sol.value(X);
plot(Xsol','o-')

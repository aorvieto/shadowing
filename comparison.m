%% clearing
clear all
close all
clc


%% all problem parameters
T = 100; %integration interval is [0,T]
mu = 0.01; %the objective function is a 1d quadratic: mu/2 x^2
eta = (1)^2; %only choose perfect squares!!
K = T/sqrt(eta); %stepsize in discretization is h, defined as square root of eta
alpha = 2*sqrt(mu); %choice of viscosity
beta = 1-sqrt(eta)*alpha; %corresponding beta
x0_1 = -2; %first point
x0_2 = 6; %second point
v0_1 = 0; %first point
v0_2 = 0; %second point

%% defining the differential
syms x(t)
Dx = diff(x);


%% Analytically solving the heavy ball ODE
ode = diff(x,t,2) + alpha*diff(x,t,1) + mu*x == 0;

% starting from (x0_1,0)
cond1 = x(0) == x0_1;
cond2 = Dx(0) == v0_1;
conds = [cond1 cond2];
xSol(t) = dsolve(ode,conds);
xSol1 = simplify(xSol);
vSol1 = simplify(diff(xSol));

% starting from (x0_2,0)
cond1 = x(0) == x0_2;
cond2 = Dx(0) == v0_2;
conds = [cond1 cond2];
xSol(t) = dsolve(ode,conds);
xSol2 = simplify(xSol);
vSol2 = simplify(diff(xSol));


% plotting norm of state in phase space
fplot(norm([xSol1;vSol1]-[xSol2;vSol2]),[0,T],'Color','b','Linewidth',2);hold on;


%% HB algorithm

%initialization
x1_HB = zeros(1,K);
v1_HB = zeros(1,K);
x2_HB = zeros(1,K);
v2_HB = zeros(1,K);
x1_HB(1) = x0_1;
x2_HB(1) = x0_2;
v1_HB(1) = v0_1;
v2_HB(1) = v0_2;

% numerical integrator (i.e. HB)
for k = 2:K
    x1_HB(k) = x1_HB(k-1) - eta*mu*x1_HB(k-1) + beta*sqrt(eta)*v1_HB(k-1);
    x2_HB(k) = x2_HB(k-1) - eta*mu*x2_HB(k-1) + beta*sqrt(eta)*v2_HB(k-1);
    v1_HB(k) = beta * v1_HB(k-1) - sqrt(eta)*mu*x1_HB(k-1);
    v2_HB(k) = beta * v2_HB(k-1) - sqrt(eta)*mu*x2_HB(k-1);
end

% plotting norm of state in phase space
plot(0:sqrt(eta):(T-sqrt(eta)),vecnorm([x1_HB;v1_HB]-[x2_HB;v2_HB]),'-o','Color','r','Linewidth',2);

legend('ODE','integrator')
title('Comparison')
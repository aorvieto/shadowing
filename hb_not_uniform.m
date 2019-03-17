clear all
close all
clc

%% defining the differential
syms x(t)
Dx = diff(x);


%% heavy ball ode
mu=0.01;
ode = diff(x,t,2) + 2*sqrt(mu)*diff(x,t,1) + mu*x == 0;

%% starting from (1,0)
cond1 = x(0) == 1;
cond2 = Dx(0) == 0;
conds = [cond1 cond2];
xSol(t) = dsolve(ode,conds);
xSol1 = simplify(xSol);
vSol1 = simplify(diff(xSol));

%% starting from (2,0)
cond1 = x(0) == 2;
cond2 = Dx(0) == 0;
conds = [cond1 cond2];
xSol(t) = dsolve(ode,conds);
xSol2 = simplify(xSol);
vSol2 = simplify(diff(xSol));


%% heavy ball contraction
fplot(norm([xSol1;vSol1]-[xSol2;vSol2]),[0,10]);hold on;

%% GD ode
ode2 = diff(x,t,1) + mu*x == 0;

%% starting from (1,0)
cond = x(0) == 1;
xSol(t) = dsolve(ode2,cond);
xSol1 = simplify(xSol);

%% starting from (2,0)
cond = x(0) == 2;
xSol(t) = dsolve(ode2,cond);
xSol2 = simplify(xSol);

%% GF contraction?
fplot(norm(xSol1-xSol2),[0,10]);hold on;




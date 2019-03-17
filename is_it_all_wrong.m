clear all
close all

syms x(t)
Dx = diff(x);
ode = diff(x,t,2) + 2*diff(x,t,1) + x == 0;
cond1 = x(0) == 1;
cond2 = Dx(0) == 0;
conds = [cond1 cond2];
xSol(t) = dsolve(ode,conds);
xSol1 = simplify(xSol);
vSol1 = simplify(diff(xSol));
%fplot(xSol,[0,1]);hold on;
%fplot(vSol,[0,1]);hold on;

cond1 = x(0) == 2;
cond2 = Dx(0) == 0;
conds = [cond1 cond2];
xSol(t) = dsolve(ode,conds);
xSol2 = simplify(xSol);
vSol2 = simplify(diff(xSol));

fplot(abs(xSol1-xSol2) ,[0,10]);hold on;




clear all
close all
clc

eta=1;
T = 100;
K = T/eta;
mu =0.1;
beta = 0.99;

x1_GD = zeros(1,K);


x1_GD(1) = 1;
x1_HB(1) = 1;

x2_GD = zeros(1,K);


x2_GD(1) = 2;
x2_HB(1) = 2;

for k = 2:K
    x1_GD(k) = x1_GD(k-1) - eta*mu*x1_GD(k-1);
    x2_GD(k) = x2_GD(k-1) - eta*mu*x2_GD(k-1);
    x1_HB(k) = x1_HB(k-1) - eta*mu*x1_HB(k-1) + beta*sqrt(eta)*v1_HB(k-1);
    x2_HB(k) = x2_HB(k-1) - eta*mu*x2_HB(k-1) + beta*sqrt(eta)*v1_HB(k-1);
    v1_HB(k) = beta * v1_HB(k-1) - sqrt(eta)*mu*x1_HB(k-1);
    v2_HB(k) = beta * v2_HB(k-1) - sqrt(eta)*mu*x2_HB(k-1);
end

plot(1:K,abs(x1_GD-x2_GD),'b');hold on
plot(1:K,vecnorm([x1_HB;v1_HB]-[x2_HB;v2_HB]),'r');
legend('GD','HB')
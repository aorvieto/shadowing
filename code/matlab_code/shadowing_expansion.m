%% Very simple script to test the shadowing in the concave case
% check http://matwbn.icm.edu.pl/ksiazki/apm/apm58/apm5833.pdf for formulas
% author: Antonio Orvieto

clear all
close all
clc


lambda = 1.2; % x |-> lambda x
epsilon = 1;
K = 30; %iterations

x = zeros(K,1);
x(1) = 1;


%% Nominal System
for i = 2:K
    x(i) = lambda*x(i-1);
end
subplot(2,2,[1,3])

plot(x,'Linewidth',2,'Color',[0.993 0.582 0.026]);hold on

%% Perturbed System
for i = 2:K
    noise = rand*2*epsilon-epsilon; %uniform in [-eps,eps]
    %noise = epsilon;
    x(i) = lambda*x(i-1)+noise;
end

plot(x,'g','Linewidth',2);hold on
x_p = x; %saving
xlabel('k','FontSize',20)


%% Shadow
%computing the initial condition y
lambda_seq = 0*x;
lambda_seq(1) = 1;
for i = 2:K
    lambda_seq(i) = lambda*lambda_seq(i-1);
end
y_seq = x./lambda_seq; %initial condition y

x(1)=y_seq(end);
for i = 2:K
    x(i) = lambda*x(i-1);
end

plot(x,'m','Linewidth',2);hold on
l=legend({'Nominal System','Perturbed System','Shadow'},'location','best')
l.set('FontSize',20);
subplot(2,2,2)
plot(y_seq,'Linewidth',2);hold on
l=legend('x_0 estimate','location','best')
l.set('FontSize',20);
xlabel('k','FontSize',20)

subplot(2,2,4)
e=plot(x_p-x,'Linewidth',2);hold on
b=plot(0*x+epsilon/(lambda-1),'Linewidth',2);hold on
plot(0*x-epsilon/(lambda-1),'m','Linewidth',2);hold on
l=legend([e,b],{'error','shadowing radius'},'location','best');
l.set('FontSize',20);
xlabel('k','FontSize',20)





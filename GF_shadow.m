%cleaning
clear all
close all
clc

%% settings simulation and plotting
T=5000; % number of iterations of 
d=2;


%% initialization and function definition
x0=[10,10]';
mu = 5*1e-3;
L = 2*1e-2;
H = [L,0;0,mu];
l = norm(H*x0);

%% Contour Plot
subplot(1,2,1)
x = -2:0.1:11;
y = -2:0.1:11;
[X,Y] = meshgrid(x,y);
Z = H(1,1)*X.^2+ H(2,2)*Y.^2;
contour(X,Y,Z,'Color',[0.5,0.5,0.5]);hold on


%% Shadowing and stepsize
epsilon_shadow = 5;
eta_simu = 1e-2;
eta_GD = epsilon_shadow*mu/(l*L);
k_eta = eta_GD/eta_simu;


%% Gradient Flow
t = 0:eta_simu:T;
x = zeros(d,length(t));
x(:,1) = x0;
for k = 1:(length(t)-1)
    x(:,k+1)=x(:,k)-eta_simu*H*x(:,k);
end

subplot(1,2,1)
hold on;
plot(x(1,:),x(2,:),'Color',[0.50,0.76,0.37],'Linewidth',2);
h1=plot(x(1,(1:end)),x(2,(1:end)),'o','Color',[0.50,0.76,0.37],'Linewidth',2);

subplot(1,2,2);
hold on;
semilogy(t/eta_GD,log(vecnorm(x,2)),'g','Linewidth',2);



%% Gradient Descent

t = 0:eta_GD:T;
x = zeros(d,length(t));
x(:,1) = x0;
for k = 1:(length(t)-1)
    x(:,k+1)=x(:,k)-eta_GD*H*x(:,k);
end

subplot(1,2,1);
hold on;
h2=plot(x(1,1:size(x,2)),x(2,1:size(x,2)),'+','Color','b','Linewidth',2);
ll=legend([h1,h2],{'$X(k\eta)$','$x_k$'},'location','best');
set(ll,'Interpreter','latex','Fontsize',16);
pbaspect([1 1 1])

subplot(1,2,2);
hold on;
plot(t/eta_GD,log(vecnorm(x,2)),'b','Linewidth',2);
ll=legend('$\log||X(k\eta)-x^*||$','$\log||x_k-x^*||$','location','best');
set(ll,'Interpreter','latex','Fontsize',16);
pbaspect([1 1 1])
grid on
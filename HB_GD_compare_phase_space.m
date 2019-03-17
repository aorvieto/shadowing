
%% Cleaning
clear all
close all
clc

%% Settings
x0=[10,10]';
mu = 5*1e-3;
L = 2*1e-2;
kappa = mu/L;
H = [L,0;0,mu];

eta = 1/(10*L);
beta = 0.85%((1-sqrt(kappa))/(1+sqrt(kappa)))^2;


%% init
iterations = 30;
x1 = zeros(2,iterations);
v1 = zeros(2,iterations);
x2 = zeros(2,iterations);
v2 = zeros(2,iterations);

x1(:,1)=x0;
x2(:,1)=x0;


for i =2:iterations
    x1(:,i) = x1(:,(i-1)) + beta*sqrt(eta)*v1(:,(i-1)) - eta*H*x1(:,(i-1));
    v1(:,i) = beta*v1(:,(i-1)) - sqrt(eta)*H*x1(:,(i-1));
    
    x2(:,i) = x2(:,(i-1)) + 0*sqrt(eta)*v2(:,(i-1)) - eta*H*x2(:,(i-1));
    v2(:,i) = 0*v2(:,(i-1)) - sqrt(eta)*H*x2(:,(i-1));
end


y = [x1,x2;v1,v2];
[U,S,V]=svd(y);
U = U(:,[1,2,3]);
S = S([1,2,3],[1,2,3]);
V = V(:,[1,2,3]);
compressed = U*S*V';


x1v1_compressed = compressed([1,2,3],1:(size(compressed,2)/2));
x2v2_compressed = compressed([1,2,3],(1+(size(compressed,2)/2)):end);

figure
subplot(1,3,[1,2])
plot3(x1v1_compressed(1,2:end)',x1v1_compressed(2,2:end)',x1v1_compressed(3,2:end)','Color',[0.50,0.76,0.37],'Linewidth',2);hold on
plot3(x2v2_compressed(1,2:end)',x2v2_compressed(2,2:end)',x2v2_compressed(3,2:end)','b','Linewidth',2);hold on
plot3(x1v1_compressed(1,2:end)',x1v1_compressed(2,2:end)',x1v1_compressed(3,2:end)','o','Color',[0.50,0.76,0.37]);hold on
plot3(x2v2_compressed(1,2:end)',x2v2_compressed(2,2:end)',x2v2_compressed(3,2:end)','o','Color','b');hold on
scatter3(0,0,0,100,'m','filled')
grid on;

text(x1v1_compressed(1,2),x1v1_compressed(2,2),x1v1_compressed(3,2),'   (x_0, v_0)','Color','k','Fontsize',14)
text(x1v1_compressed(1,3),x1v1_compressed(2,3),x1v1_compressed(3,3),'     Heavy Ball','Color',[0.50,0.76,0.37],'Fontsize',14)
text(x2v2_compressed(1,5),x2v2_compressed(2,5),x2v2_compressed(3,5)+0.3,'Gradient Descent','Color','b','Fontsize',14)
text(x2v2_compressed(1,end)-2,x1v1_compressed(2,end)+2,x1v1_compressed(3,end),'   (x*,v*)','Color','m','Fontsize',14)
title('projected phase space (position, momentum)')


subplot(1,3,3)
hold on
plot(x1(1,:),x1(2,:),'o','Color',[0.50,0.76,0.37]);
plot(x1(1,:),x1(2,:),'-','Color',[0.50,0.76,0.37],'Linewidth',2);
plot(x2(1,:),x2(2,:),'o','Color','b');
plot(x2(1,:),x2(2,:),'-','Color','b','Linewidth',2);
scatter3(0,0,0,100,'m','filled')
xlim([-5,12])
ylim([-5,12])
grid on
text(x1(1,2),x1(2,2),'       x_0','Color','k','Fontsize',14)
text(x1(1,3)-8,x1(2,3)-0.5,'Heavy Ball','Color',[0.50,0.76,0.37],'Fontsize',14)
text(x2(1,5)-3,x2(2,5)-3,'Gradient Descent','Color','b','Fontsize',14)
text(x2(1,end),x1(2,end)+1,'   x*','Color','m','Fontsize',14)
title('original space (position)')
% semilogy(1:iterations, vecnorm(x1v1_compressed),'Linewidth',2,'Color',[0.50,0.76,0.37]);hold on
% semilogy(1:iterations, vecnorm(x2v2_compressed),'Linewidth',2,'Color','b');
% legend('HB','GD')
% xlabel('Iterations')
% ylabel('Distance to solution (phase space)')
% grid on

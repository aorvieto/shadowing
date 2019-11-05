%% SHADOWING IN LINEAR REGRESSION (GD)
% author: Antonio Orvieto

clear all
close all
clc

%% Definition of the problem
n = 100; %number of datapoints
d = 5; % problem dimension
sigma = 0.1; %linear regression variance
x0 = randn(d,1); %x0

%% Synthetic dataset
x_star = randn(d,1); %solution to the problem
A = randn(n,d); %random data points
y = A*x_star+sigma*randn(n,1); %target variable
x_sol = A\y; %getting the numerical correct solution to the problem
L=max(eig(A'*A)); %getting smoothness constant
mu=min(eig(A'*A)); %getting strong convexity constant

%% ODE and Shadowing tuning parameters
epsilon = 0.001; %desired tracking accuracy
h_ODE = 1e-5; %integration interval
nit_ODE = 1e6; %number of iterations of numerical integration
x_ODE=zeros(d,nit_ODE);
f_ODE=zeros(1,nit_ODE);
nabla_f_ODE=zeros(1,nit_ODE);
perturbation = randn(size(x0));
perturbation = epsilon*perturbation./vecnorm(perturbation);
x_ODE(:,1) = x0+0*perturbation;
f_ODE(1) = (1/n)*norm(A*x_ODE(:,1)-y)^2-(1/n)*norm(A*x_sol-y)^2;
nabla_f_ODE(1)=vecnorm((2/n)*A'*(A*x_ODE(:,1)-y));

%% running with GD-ODE
for i = 2:nit_ODE
    %GD-ODE
    grad_f = (2/n)*A'*(A*x_ODE(:,i-1)-y); 
    nabla_f_ODE(i) = vecnorm(grad_f);
    x_ODE(:,i) = x_ODE(:,i-1)-h_ODE*grad_f;
    f_ODE(i) = (1/n)*norm(A*x_ODE(:,i)-y)^2-(1/n)*norm(A*x_sol-y)^2;
end 

%% computation shadowing stepsize and initialization
%computation of shadowing stepsize
ell = max(nabla_f_ODE); %gradient norm at initial position is the biggest
h= min(2*epsilon*mu/(L*ell),1/L); %theoretical learning rate
h = (floor(h/h_ODE))*h_ODE; %making stepsize a multiple of ODE step
ratio_h = round(h/h_ODE);
nit = ceil(nit_ODE/ratio_h); % number or iterations

%initialization of GD
x=zeros(d,nit);
f=zeros(1,nit);
x(:,1) = x0;
f(1,1) = (1/n)*norm(A*x(:,1)-y)^2-(1/n)*norm(A*x_sol-y)^2;



%% learning with GD
for i = 2:nit
    %GD
    grad_f = (2/n)*A'*(A*x(:,i-1)-y); 
    x(:,i) = x(:,i-1)-h*grad_f;
    f(:,i) = (1/n)*norm(A*x(:,i)-y)^2-(1/n)*norm(A*x_sol-y)^2;
end 



%% subsampling GD-ODE
f_ODE = f_ODE(1:ratio_h:end);
x_ODE = x_ODE(:,1:ratio_h:end);

%% plotting
figure(1)
subplot(1,4,1)
h2=semilogy(1:nit,abs(f_ODE),'Linewidth',2,'Color',[0.181 0.702 1.0]);hold on
h1=semilogy(1:nit,abs(f),'--','Linewidth',2,'Color',[0.993 0.582 0.026]);hold on
xlabel('$k$','FontSize',18,'Interpreter','Latex')
ylabel('suboptimality','Fontsize',18,'Interpreter','Latex')
l=legend([h1,h2],{'GD','GD-ODE'},'Interpreter','Latex');
xlim([1,nit]);
l.FontSize = 20;
subplot(1,4,2)
plot(1:nit,vecnorm(x-x_ODE,2),'Linewidth',2,'Color',[0.278 0.71 0.184]);hold on;
plot(1:nit,0*(1:nit)+epsilon,'--','Linewidth',2,'Color','k');hold on;
xlabel('$k$','FontSize',18,'Interpreter','Latex')
ylabel('$\|x_k - y_k\|$','FontSize',20,'Interpreter','Latex')
text(20,1.1*epsilon,'bound: $\epsilon = \frac{h\ell L}{2\mu}$','FontSize',18,'Interpreter','Latex')
ylim([0,1.2*epsilon])
xlim([1,nit]);

%%plotting 
loadings = pca(x_ODE'-x');
principal_components = loadings(:,1:3);
x_ODE_compressed = (x_ODE'*principal_components);
x_compressed = (x'*principal_components);
x0_compressed = (x0'*principal_components);
x_sol_compressed = (x_sol'*principal_components);

subplot(1,4,[3,4])
h11=plot3(x_ODE_compressed(:,1),x_ODE_compressed(:,2),x_ODE_compressed(:,3),'Linewidth',2,'Color',[0.181 0.702 1.0]);hold on
h22=plot3(x_compressed(:,1),x_compressed(:,2),x_compressed(:,3),'--','Linewidth',2,'Color',[0.993 0.582 0.026]);
grid on;
h4=plot3(x0_compressed(1),x0_compressed(2),x0_compressed(3),'-o','Color','k','MarkerFaceColor','k','MarkerSize', 5);
h5=plot3(x_sol_compressed(1),x_sol_compressed(2),x_sol_compressed(3),'-kh','MarkerSize', 20,'MarkerFaceColor',[0.839 0.78 0.078]);
l=legend([h5,h4],{'solution','$x_0$'},'Fontsize',20,'Interpreter','Latex');
xlabel('PC 1','FontSize',14,'Interpreter','Latex')
ylabel('PC 2','FontSize',14,'Interpreter','Latex')
zlabel('PC 3','FontSize',14,'Interpreter','Latex')
set(gcf, 'Units', 'Normalized', 'OuterPosition', [0, 0.04, 1, 0.96]);

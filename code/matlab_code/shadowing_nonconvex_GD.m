%% SHADOWING IN NON-CONVEX OBJECTIVES (GD)
% author: Antonio Orvieto

clear all
close all
clc

%% settings
set(0, 'DefaultFigureRenderer', 'painters')
tf = 10;
eta_ODE = 1e-5;
eta = 0.3;
ratio_eta=round(eta/eta_ODE);

n = round(tf/eta)+1;
n_ODE = round(tf/eta_ODE)+1;
d = 2;

x0 = [2.05;2.5];

addpath(genpath('test_functions'));

countour_f=@hozaki_contour;
f=@hozaki;
gradf=@grad_hozaki;

%% init
x = zeros(d,n);
x_ODE = zeros(d,n_ODE);
x(:,1) = x0 +[0.0225;0];
x_ODE(:,1) = x0;

%% optimization

for k = 2:n_ODE
    x_ODE(:,k) = x_ODE(:,k-1)-eta_ODE*gradf(x_ODE(:,k-1));
end

for k = 2:n
    x(:,k) = x(:,k-1)-eta*gradf(x(:,k-1));
end

%% plotting
if d==2
    subplot(1,3,[1,2])
    %contour
    XX=-0:0.01:5;
    YY=XX;
    [X,Y] = meshgrid(XX,YY);
    
    Z = countour_f(X,Y);
    minZ = min(min(Z));
    [idx,idy] = find(Z==minZ);
    contour(X,Y,Z,30,'Color',[0.85,0.85,0.85],'Linewidth',2);hold on;
    
    %solution
    h5=plot(Y(idy,idy),X(idx,idx),'-kh','MarkerSize', 20,'MarkerFaceColor',[0.839 0.78 0.078]);
    
    %trajectories
    plot(x(1,:),x(2,:),'Color',[0.993 0.582 0.026],'Linewidth',1);hold on
    h1=plot(x(1,:),x(2,:),'o','Color',[0.993 0.582 0.026],'Linewidth',2, 'MarkerFaceColor', [0.993 0.582 0.026]);hold on
    plot(x_ODE(1,:),x_ODE(2,:),'Color',[0.181 0.702 1.0],'Linewidth',1);hold on
    h2=plot(x_ODE(1,1:ratio_eta:end),x_ODE(2,1:ratio_eta:end),'o','Color',[0.181 0.702 1.0],'Linewidth',2, 'MarkerFaceColor',[0.181 0.702 1.0]);hold on
    hsp1 = get(gca, 'Position');
    plot(3+0*XX, XX,'--','Color','k','Linewidth',2);hold on
    l2=legend([h1,h2,h5],{'Gradient Descent: $x_k$','GD-ODE: $ \ \ \ \ \ \ \ \ \ \ \ y_k$','solution'},'Location','NorthEast','Interpreter','Latex');
    l2.set('FontSize',20);
    text(x0(1)-0.5,x0(2)+0.25,'$\ x_0 \ne y_0$','FontSize',25,'Interpreter','Latex')
    text(x0(1)-0.5,x0(2)-1.2,'Thm. 4','FontSize',25,'Interpreter','Latex')
    text(x0(1)+1.3,x0(2)-1.2,'Thm. 3','FontSize',25,'Interpreter','Latex')

    xlim([1.4,5])
    ylim([0.9,3.6])

    subplot(1,3,3)
    plot((1:size(x,2))-1,vecnorm(x-x_ODE(:,1:ratio_eta:end),2),'-+','Linewidth',2,'Color',[0.278 0.71 0.184]); hold on;
    hsp2 = get(gca, 'Position');  
    set(gca, 'Position', [hsp1(1)+ 1.2*hsp1(3),hsp1(2)+ 0.2*hsp1(4),  hsp1(3)/2, 0.6*hsp1(4)])
    xlabel('$k$','FontSize',18,'Interpreter','Latex')
    ylabel('$\|x_k - y_k\|$','FontSize',20,'Interpreter','Latex')
    grid on
end
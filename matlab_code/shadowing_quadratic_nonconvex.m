%% clearing
clear all
close all
clc

%% settings
set(0, 'DefaultFigureRenderer', 'painters')
tf = 1.4;
eta_ODE = 1e-5;
eta = 0.2;
ratio_eta=round(eta/eta_ODE);

n = round(tf/eta)+1;
n_ODE = round(tf/eta_ODE)+1;
d = 2;

x0 = [2;4.5];

%% init
x = zeros(d,n);
x_ODE = zeros(d,n_ODE);
x(:,1) = x0;
x_ODE(:,1) = x0;

mu = -1.27; L = 2;
H = [mu,0;0,L];

%% optimization

for k = 2:n_ODE
    x_ODE(:,k) = x_ODE(:,k-1)-eta_ODE*H*x_ODE(:,k-1);
end

%go to shadowing orbit of GD
%x(1,1)=(1-mu*eta)^(-size(x,2)+1) * x_ODE(1,end);


for k = 2:n
    x(:,k) = x(:,k-1)-eta*H*x(:,k-1);
end

%% plotting
if d==2
    subplot(1,3,[1,2])
    %contour
    XX=-2:0.01:12;
    YY=-3:0.01:9;
    [X,Y] = meshgrid(XX,YY);
    
    Z = H(1,1)*X.^2 + H(2,2)*Y.^2;
    minZ = min(min(Z));
    [idx,idy] = find(Z==minZ);
    contour(X,Y,Z,30,'Color',[0.85,0.85,0.85],'Linewidth',2);hold on;
        
    %trajectories
    plot(x(1,:),x(2,:),'Color',[0.993 0.582 0.026],'Linewidth',2);hold on
    h1=plot(x(1,:),x(2,:),'o',v,'Linewidth',2, 'MarkerFaceColor', [0.993 0.582 0.026]);hold on
    plot(x_ODE(1,:),x_ODE(2,:),'Color',[0.181 0.702 1.0],'Linewidth',2);hold on
    h2=plot(x_ODE(1,1:ratio_eta:end),x_ODE(2,1:ratio_eta:end),'o','Color',[0.181 0.702 1.0],'Linewidth',2, 'MarkerFaceColor',[0.181 0.702 1.0]);hold on
    hsp1 = get(gca, 'Position');
    
    l2=legend([h1,h2],{'Gradient Descent: $x_k$','GD-ODE: $ \ \ \ \ \ \ \ \ \ \ \ y_k$'},'Location','NorthEast');
    l2.set('FontSize',18,'Interpreter','Latex');
    text(x0(1),x0(2)+0.8,'$\ x_0 = y_0$','FontSize',18,'Interpreter','Latex')
    xlim([XX(1),XX(end)]);
    ylim([YY(1),YY(end)]);
    axis equal
    %text(x_ODE(1,end),x_ODE(2,end),'$\ \ x_\infty$','FontSize',25,'Interpreter','Latex')
    subplot(1,3,3)
    predicted = (eta*L^2*norm(x0))/(2*mu);
    plot((1:size(x,2))-1,vecnorm(x-x_ODE(:,1:ratio_eta:end),2),'-+','Linewidth',2,'Color',[0.278 0.71 0.184]); hold on;
    %plot((1:size(x,2))-1,predicted+0*vecnorm(x-x_ODE(:,1:ratio_eta:end),2),'--','Linewidth',2,'Color','k') 
    hsp2 = get(gca, 'Position');  
    set(gca, 'Position', [hsp1(1)+ 1.2*hsp1(3),hsp1(2)+ 0.2*hsp1(4),  hsp1(3)/2, 0.6*hsp1(4)])
    xlabel('$k$','FontSize',18,'Interpreter','Latex')
    ylabel('$\|x_k - y_k\|$','FontSize',20,'Interpreter','Latex')
    ylim([0,2.25])
    %l=legend('Actual');
    %l.set('FontSize',18,'Interpreter','Latex','location','North');
    grid on
end

clear all
close all
clc

T=500;
kkk=3; % plot each kkk data point
d=2;



x0=[10,10]';
mu = 5*1e-3;
L = 2*1e-2;
H = [L,0;0,mu];
l = norm(H*x0);

epsilon_shadow = 0.5;



%% Contour Plot
x = -2:0.1:11;
y = -2:0.1:11;
[X,Y] = meshgrid(x,y);
Z = H(1,1)*X.^2+ H(2,2)*Y.^2;
contour(X,Y,Z,'Color',[0.5,0.5,0.5]);hold on



% GF
dt = 1e-3;
t = 0:dt:T;
x = zeros(d,length(t));
M = zeros(d,length(t));
x(:,1) = x0;
for k = 1:(length(t)-1)
    x(:,k+1)=x(:,k)-dt*(H*x(:,k));
end

% plotting the shadow bound
for i = 1:400:length(x(1,:))
    h0=circle(x(1,i),x(2,i),epsilon_shadow); hold on
end

hold on;
h1=plot(x(1,:),x(2,:),'m','Linewidth',2);



% GD
x0=[10,10+epsilon_shadow]';
bench = 2*epsilon_shadow*mu/(l*L);

%dt = epsilon_shadow*mu/(l*L)
dt = bench;

t = 0:dt:T;
x = zeros(d,length(t));
x(:,1) = x0;
for k = 1:(length(t)-1)
    x(:,k+1)=x(:,k)-dt*(H*x(:,k));
end

hold on;
h2=plot(x(1,1:kkk:size(x,2)),x(2,1:kkk:size(x,2)),'+','Color','k');
l=legend([h2,h1,h0,],{['GD, \eta = ',num2str(dt)],'GF solution',['\epsilon - Shadow, \epsilon = ',num2str(epsilon_shadow)]},'location','best');
l.FontSize=16;

function h = circle(x,y,r)
hold on
th = 0:pi/50:2*pi;
xunit = r * cos(th) + x;
yunit = r * sin(th) + y;
h=fill(xunit,yunit,'c','LineStyle','none');
hold off;
end


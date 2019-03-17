clear all
close all
clc

T=1;
kkk=1; % plot each kkk data point
d=2;
x0=[10,10]';
H = [2*1e-2,0;0,5*1e-3];


% solution 1
dt = 5;
t = 0:dt:T;
x = zeros(d,length(t));
x(:,1) = x0;
for k = 1:(length(t)-1)
    x(:,k+1)=x(:,k)-dt*(H*x(:,k));
end

hold on;
plot(x(1,1:kkk:size(x,2)),x(2,1:kkk:size(x,2)),'+');


% % solution 2
% dt = 0.25;
% t = 0:dt:T;
% x = zeros(d,length(t));
% x(:,1) = x0;
% for k = 1:(length(t)-1)
%     x(:,k+1)=x(:,k)-dt*(H*x(:,k));
% end
% 
% hold on;
% plot(x(1,1:kkk:size(x,2)),x(2,1:kkk:size(x,2)),'o');


% super accurate solution
dt = 1e-3;
t = 0:dt:T;
x = zeros(d,length(t));
M = zeros(d,length(t));
x(:,1) = x0;
for k = 1:(length(t)-1)
    x(:,k+1)=x(:,k)-dt*(H*x(:,k));
    if k>1
        M1(k) = (x(1,k+1)-2*x(1,k)+x(1,k-1))/(dt^2);
        M2(k) = (x(2,k+1)-2*x(2,k)+x(2,k-1))/(dt^2);
    end
end

hold on;
plot(x(1,:),x(2,:),'k');

%legend('GD, \eta = 2','GD, \eta = 1','GF')




figure
plot(M1);hold on
plot(M2)

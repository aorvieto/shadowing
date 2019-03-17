clear all
close all
clc

T=1500;
kkk=1;
d=2;
x0=[0.1,0.1]';
H = [2*1e-2,0;0,5*1e-3];
alpha = 0.1;

%discretization 1
sqrt_delta = 5;
t = 0:sqrt_delta:T;
x = zeros(d,length(t));
x(:,1) = x0;

eta = (sqrt_delta^2)/(1+alpha*sqrt_delta/2)
beta = (2-alpha*sqrt_delta)/(2+alpha*sqrt_delta)

for k = 1:(length(t)-1)
    if k == 1
        x(:,k+1)=x(:,k)-eta*(H*x(:,k));
    else
        x(:,k+1)=x(:,k)+beta*(x(:,k)-x(:,k-1))-eta*(H*x(:,k));
    end        
end

hold on;
plot(x(1,1:kkk:size(x,2)),x(2,1:kkk:size(x,2)),'o');

%discretization 2
sqrt_delta = 1;
t = 0:sqrt_delta:T;
x = zeros(d,length(t));
x(:,1) = x0;

eta = (sqrt_delta^2)/(1+alpha*sqrt_delta/2)
beta = (2-alpha*sqrt_delta)/(2+alpha*sqrt_delta)

for k = 1:(length(t)-1)
    if k == 1
        x(:,k+1)=x(:,k)-eta*(H*x(:,k));
    else
        x(:,k+1)=x(:,k)+beta*(x(:,k)-x(:,k-1))-eta*(H*x(:,k));
    end        
end

plot(x(1,1:kkk:size(x,2)),x(2,1:kkk:size(x,2)),'o');


%discretization 3
sqrt_delta = 0.05;
t = 0:sqrt_delta:T;
x = zeros(d,length(t));
x(:,1) = x0;

eta = (sqrt_delta^2)/(1+alpha*sqrt_delta/2)
beta = (2-alpha*sqrt_delta)/(2+alpha*sqrt_delta)

for k = 1:(length(t)-1)
    if k == 1
        x(:,k+1)=x(:,k)-eta*(H*x(:,k));
    else
        x(:,k+1)=x(:,k)+beta*(x(:,k)-x(:,k-1))-eta*(H*x(:,k));
    end        
end


hold on;
plot(x(1,1:kkk:size(x,2)),x(2,1:kkk:size(x,2)),'k');




legend('HB, \delta^{1/2} = 5','HB, \delta^{1/2} = 1','HBF')


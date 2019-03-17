clear all
close all
clc

lambda = 5;
beta=0:0.001:1;
eta = 0.1;

new_eig_1 = zeros(size(beta));
new_eig_2 = zeros(size(beta));

for i = 1:length(beta)
    new_eig_1(i)=abs(0.5*(beta(i)+1-eta*lambda+sqrt((beta(i)+1-eta*lambda)^2-4*beta(i))));
    new_eig_2(i)=abs(0.5*(beta(i)+1-eta*lambda-sqrt((beta(i)+1-eta*lambda)^2-4*beta(i))));
end

plot(beta,new_eig_1,'k'); hold on;
plot(beta,new_eig_2,'k');
plot(beta,sqrt(beta));
plot(beta,0*beta+1-eta*lambda)
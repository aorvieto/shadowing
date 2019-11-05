%% Script in support for the proof of hyperbolicity of HB
% author: Antonio Orvieto

clear all
close all

beta = 0.001:0.001:2;
h = 1;
lambda = 2;
roots_hist = zeros(2,length(beta));


for i= 1:length(beta)
    roots_hist(:,i)=roots([1, -(1+beta(i)-h^2*lambda), beta(i)]);
    roots_hist(1,i) = abs(roots_hist(1,i));
    roots_hist(2,i) = abs(roots_hist(2,i));
end
plot(beta,roots_hist(1,:));hold on
plot(beta,roots_hist(2,:))

figure
for i= 1:length(beta)
    roots_hist(:,i)=roots([1, -(1+beta(i)-(1-beta(i))*h^2*lambda), beta(i)]);
    roots_hist(1,i) = abs(roots_hist(1,i));
    roots_hist(2,i) = abs(roots_hist(2,i));
end
plot(beta,roots_hist(1,:));hold on
plot(beta,roots_hist(2,:))
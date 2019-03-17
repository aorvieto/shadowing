clear all
close all
clc
lambda = 0.1;
alpha = 0:0.01:1;
h=0.01;

new_eig_1 = zeros(size(alpha));
new_eig_2 = zeros(size(alpha));

for i = 1:length(alpha)
   r=roots([1,alpha(i),lambda]);
   new_eig_1(i) = r(1);
   new_eig_2(i) = r(2);   
end

plot(alpha,real(new_eig_1));hold on
plot(alpha,real(new_eig_2));

lambda = 0.2;

for i = 1:length(alpha)
   r=roots([1,alpha(i),lambda]);
   new_eig_1(i) = r(1);
   new_eig_2(i) = r(2);   
end

plot(alpha,real(new_eig_1));hold on
plot(alpha,real(new_eig_2));



clear all
close all
rho_1 =0.99;
rho_2 =0.99001;
delta = 5;

K=1000;
shadow_rad = zeros(1,K);
shadow_rad_pred = zeros(1,K);


for k=1:K
    for i = 0:k
        shadow_rad(k) = shadow_rad(k)+(rho_1^(k-i)*rho_2^i);
    end
    shadow_rad_pred(k) = (rho_2^(k+1)-rho_1^(k+1))/(rho_2-rho_1);
end

plot(1:K,shadow_rad);hold on
plot(1:K,(1:K)*0+(1/(1-rho_1)))
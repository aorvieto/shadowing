function gradf=grad_hozaki(x) %input is d x n
    gradf = [0;0];
    gradf(1) = (-8+14*x(1)-7*x(1).^2+x(1)^3)*(x(2)^2)*exp(-x(2));
    gradf(2) = (1-8*x(1)+7*x(1)^2-(7/3)*x(1)^3+x(1)^4/4)*(2*x(2)*exp(-x(2))-x(2)^2.*exp(-x(2)));
end
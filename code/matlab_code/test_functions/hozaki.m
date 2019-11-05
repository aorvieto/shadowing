function y=hozaki(x) %input is d x n
    y = (1-8*x(1)+7*x(1).^2-(7/3)*x(1).^3+x(1).^4/4).*x(2).^2.*exp(-x(2));
end
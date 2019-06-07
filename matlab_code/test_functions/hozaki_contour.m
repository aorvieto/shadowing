function Z= hozaki_contour(X,Y)
    Z=(1-8*X+7*X.^2-(7/3)*X.^3+(1/4)*X.^4).*(Y.^2).*exp(-Y);
end
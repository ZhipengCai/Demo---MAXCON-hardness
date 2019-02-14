function lineplot(h,a,par,clr)

m = par(1);
c = par(2);

figure(h);
x = linspace(a(1),a(2),100);
y = m*x + repmat(c,1,100);
plot(x,y,'-','color',clr, 'LineWidth', 1.2);
end
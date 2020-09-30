
%% Example 2

clear;close all;

a  = -5:0.02:1;
bp = 0.01:0.02:5;

[X,Y] = meshgrid(a,bp);
Z = X./Y;
figure;
s = surf(X,Y,Z,'FaceAlpha',0.6);
s.EdgeColor = 'none';
hold on;

[X1,Y1] = meshgrid(a,-bp);
Z1 = X1./Y1;
s = surf(X1,Y1,Z1,'FaceAlpha',0.6);
s.EdgeColor = 'none';

%xlim([0,2]);ylim([0,2])
zlim([-50,50])
ylabel('$B_{K}$','Interpreter','latex','FontSize',10);
xlabel('$A_{K}$','Interpreter','latex','FontSize',10);
zlabel('$C_{K}$','Interpreter','latex','FontSize',10);
set(gcf,'Position',[250 150 300 300]);
set(gca,'TickLabelInterpreter','latex')
view(-110,20)
print(gcf,'Fig2a','-painters','-depsc','-r600')

%%  
a = 0.5;
v = [-5:0.1:-0.1];  % plotting range from -5 to 5
y1 = a./v;
y2 = a./(-v);

y3 = max(y1).*ones(length(v),1);
y4 = min(y2).*ones(length(v),1);

figure;
plot(v,y1,'k:'); hold on
plot(-v,y2,'k:')
ylim([-8,12])
ax = gca;
ax.XAxisLocation = 'origin';
ax.YAxisLocation = 'origin';
box off


set(gca,'TickLabelInterpreter','latex','fontsize',12)
%set(gca,'XTickLabel',{'-5','-3','-1','1','3','5'},'XTick',[-5:2:5]);

set(gca, 'Layer', 'top');

% lables
text(8,1,'$$B_K$$','Interpreter','latex','FontSize',10)
text(0.5,9,'$$C_K$$','Interpreter','latex','FontSize',10)

% fill in region

% hold on
y = -8:0.05:10;
x = -8:0.05:10;
[k2, k1] = meshgrid(x,y);  % get 2-D mesh for x and y
conditions = (k1.*k2 > a);
cond = zeros(length(y),length(x))-100; % Initialize
cond(conditions) = nan;
s = surf(k1, k2, cond, 'FaceAlpha',0.6);
s.EdgeColor = 'none';

xlim([-8,10]);ylim([-8,10]);

set(gcf,'Position',[250 150 300 300]);
set(gca,'TickLabelInterpreter','latex')
%print(gcf,'Fig2b.eps','-painters','-depsc','-r 600')
print(gcf,'Fig2b','-painters','-dpng','-r 600')








%% Example 2

clear;close all;

A = -1; B = 1; C = 1;
Q = 1; R = 1;
V = 1; W = 1;


Ak = 1 - 2*sqrt(2);

%Ak = -2;
%% compute LQG cost

Num = 250;

J1 = zeros(Num,Num);

b1 = linspace(-5,5,Num);
c1 = linspace(-5,5,Num);
for i = 1:Num
    for j = 1:Num
    
    Bk = b1(i);
    Ck = c1(j);
    hA = [A B*Ck;Bk*C Ak];
    if max(real(eig(hA))) < 0
        X = lyap(hA,blkdiag(W,Bk*V*Bk'));
        J1(i,j) = trace(blkdiag(Q,Ck'*R*Ck')*X);
    else
        J1(i,j) = inf;
    end
    end
end

% J2 = zeros(Num,Num);
% b2 = linspace(0,5,Num);
% c2 = linspace(-5,0,Num);
% for i = 1:Num
%     for j = 1:Num
%     
%     Bk = b2(i);
%     Ck = c2(j);
%     hA = [A B*Ck;Bk*C Ak];
%     if max(real(eig(hA))) < 0
%         X = lyap(hA,blkdiag(W,Bk*V*Bk'));
%         J2(i,j) = trace(blkdiag(Q,Ck'*R*Ck')*X);
%     else
%         J2(i,j) = inf;
%     end
%     end
% end


% plot

figure
[X1,Y1] = meshgrid(b1,c1);
s = surf(X1,Y1,J1,'FaceAlpha',0.6);
s.EdgeColor = 'none';
hold on;
zlim([0 100]);

% [X2,Y2] = meshgrid(b2,c2);
% s = surf(X2,Y2,J2,'FaceAlpha',0.6);
% s.EdgeColor = 'none';
% zlim([0,500]);

Bk = -1+sqrt(2);
Ck = 1 - sqrt(2);
hA = [A B*Ck;Bk*C Ak];
if max(real(eig(hA))) < 0
    X = lyap(hA,blkdiag(W,Bk*V*Bk'));
    z = trace(blkdiag(Q,Ck'*R*Ck')*X);   % same with the value below
end

z1 = min(min(J1));

Num = 100;
T = linspace(0.05,5,Num);
x = Bk./T;
y = Ck.*T;
z = ones(Num,1)*z1;
plot3(x,y,z,'r','linewidth',1.5);

%T = linspace(0.05,5,Num);
x = -Bk./T;
y = -Ck.*T;
z = ones(Num,1)*z1;
plot3(x,y,z,'r','linewidth',1.5);

xlim([-5,5]);ylim([-5,5]);

set(gcf,'Position',[250 150 400 350]);
ylabel('$C_{K}$','Interpreter','latex','FontSize',10);
xlabel('$B_{K}$','Interpreter','latex','FontSize',10);
zlabel('$J_n$','Interpreter','latex','FontSize',10);
set(gca,'TickLabelInterpreter','latex')
view(80,20)
print(gcf,'Fig4','-painters','-depsc','-r600')





% -----------------------------------------------------------------------
% Example From Mario Sznaier
% Authors: Yand Zheng, Yujie Tang, Na Li
% Title:   Analysis of the Optimization Landscape of 
%                                        Linear Quadratic Gaussian Control
%-------------------------------------------------------------------------

addpath('../lqg_utils')
addpath('..');

clc;clear; close all

% dynamics
s = tf('s');
G = (s-1)/s/(s-2);

Gs = ss(G);  % state-space model
A = Gs.A; B = Gs.B; C = Gs.C;
nx = size(A,1);
ny = size(C,1);
nu = size(B,2);

Qc =eye(2);
R = 1;
W = eye(2);
V = 1;

% ---------------------------------------------
% Globally optimal controller
% ---------------------------------------------
S = are(A,B*R^(-1)*B',Qc);
K = lqr(A,B,Qc,R);
P = are(A',C'*V^(-1)*C,W);
L = P*C'*V^(-1);

% dynamic controller
Ak = A - B*K- L*C;
Bk = L;
Ck = -K;

Ko.Ak = Ak; Ko.Bk = Bk;Ko.Ck = Ck;
% cost
hA    = [A B*Ck;Bk*C Ak];
Y     = lyap(hA',blkdiag(Qc,Ck'*R*Ck));
Jopt  = trace(blkdiag(W,Bk*Bk')*Y);

Num = 4; % number of initial points
% ---------------------------------------------
% Gradient descent 
% ---------------------------------------------
info_full = cell(Num,1);
info_cano = cell(Num,1);
K_full    = cell(Num,1);  % controller
K_cano    = cell(Num,1);  % controller
for ind = 1:Num
%     gamma = 1e-2;
%     count = 1;
%     while true   % generating an initial point that is close to globally optimal solution
%         Ak = Ko.Ak + gamma*randn(nx,nx); 
%         Bk = Ko.Bk + gamma*randn(nx,ny); 
%         Ck = Ko.Ck + gamma*randn(nu,nx);
%         K0.Ak = Ak; K0.Bk = Bk;K0.Ck = Ck;
%         Acl  = [A B*K0.Ck; K0.Bk*C K0.Ak];
%         if  max(real(eig(Acl))) < 0
%             max(real(eig(Acl)))
%             count
%             break
%         end
%         count = count + 1
%     end

    p  = rand(nx,1)-2;   % a random number between (-2,-1)
    K  = place(A,B,p);
    L  = place(A',C',p); L = L';
    Ak = A - B*K- L*C;
    Bk = L;
    Ck = -K;
    K0.Ak = Ak; K0.Bk = Bk;K0.Ck = Ck;


    opts.tol      = 1e-4;
    opts.stepsize = 1e-3;
    opts.maxIter  = 1e3;
    opts.Disp     = 100;

    % full gradient
    opts.stepsize = 1e0;
    [K1,J1,info1] = LQG_gd_full(A,B,C,Qc,R,W,V,K0,opts);
    info_full{ind} = info1;
    K_full{ind}    = K1;
    % Hess1 = LQGhessfull(A,B,C,K1,Qc,R,W,V);  % hessian

    % gradient over canonical form
    opts.stepsize = 1e0;
    [K2,J2,info2] = LQG_gd_cano(A,B,C,Qc,R,W,V,K0,opts);
    info_cano{ind} = info2;
    K_cano{ind}    = K2;
    % Hess2 = LQGhessfull(A,B,C,K2,Qc,R,W,V);  % hessian
end

% ------------------------------------------------------------------------
%          Figure
% ------------------------------------------------------------------------
lineWidth = 1.5;
colorName = {'b','g','m','k'};

figure;
for ind = 1:Num
    index = 1:1:info_full{ind}.iter;
    semilogy(index,(info_full{ind}.Jiter(index)-Jopt)/Jopt,colorName{ind},'linewidth',lineWidth); hold on;
end
ylabel('Suboptimality $(J(K) - J^*)/J^*$','Interpreter','latex','FontSize',10);
xlabel('Iterations $t$','Interpreter','latex','FontSize',10);
set(gcf,'Position',[250 150 300 300]);
set(gca,'TickLabelInterpreter','latex')
print(gcf,'Fig_mario_ex1_1','-painters','-dpng','-r 600')

figure;
for ind = 1:Num
    index = 1:1:info_cano{ind}.iter;
    semilogy(index,(info_cano{ind}.Jiter(index)-Jopt)/Jopt,colorName{ind},'linewidth',lineWidth); hold on;
end
ylabel('Suboptimality $(J(K) - J^*)/J^*$','Interpreter','latex','FontSize',10);
xlabel('Iterations $t$','Interpreter','latex','FontSize',10);
set(gcf,'Position',[250 150 300 300]);
set(gca,'TickLabelInterpreter','latex')
print(gcf,'Fig_mario_ex1_2','-painters','-dpng','-r 600')

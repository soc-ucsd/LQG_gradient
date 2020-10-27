% -----------------------------------------------------------------------
% Example 8 in the Paper
% Authors: Yand Zheng, Yujie Tang, Na Li
% Title:   Analysis of the Optimization Landscape of 
%                                        Linear Quadratic Gaussian Control
%-------------------------------------------------------------------------

addpath('../lqg_utils')
addpath('..');

clc;clear; close all

nx = 2;    % Number of states
nu = 1;    % Number of inputs;
ny = 1;    % Number of outputs

epsilon = 0.2;
% dynamics
A = 3/2*[-1 0;0 -1-epsilon];
B = [1;1+epsilon];
C = [1,1];

Qc = [4 1;1 4];
R = 1;
W = [4 1+epsilon;1+epsilon 4*(1+epsilon)^2];
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
    p = rand(2,1)-2;   % a random number between (-2,-1)
    K = place(A,B,p);
    L = place(A',C',p); L = L';
    Ak = A - B*K- L*C;
    Bk = L;
    Ck = -K;
    %Ak = -50*eye(2); Bk = 50*rand(nx,ny); Ck = zeros(nu,nx);
    K0.Ak = Ak; K0.Bk = Bk;K0.Ck = Ck;

    opts.tol      = 1e-6;
    opts.maxIter  = 1e4;

    % full gradient
    opts.stepsize = 1e0;
    [K1,J1,info1] = LQGgd(A,B,C,Qc,R,W,V,K0,opts);
    info_full{ind} = info1;
    K_full{ind}    = K1;
    % Hess1 = LQGhessfull(A,B,C,K1,Qc,R,W,V);  % hessian

    % gradient over canonical form
    opts.stepsize = 1e0;
    [K2,J2,info2] = LQGgd_can(A,B,C,Qc,R,W,V,K0,opts);
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
    semilogy(index,info_full{ind}.Jiter(index)-Jopt,colorName{ind},'linewidth',lineWidth); hold on;
end
ylabel('Suboptimality ($J(K) - J^*$)','Interpreter','latex','FontSize',10);
xlabel('Iterations $t$','Interpreter','latex','FontSize',10);
set(gcf,'Position',[250 150 300 300]);
set(gca,'TickLabelInterpreter','latex')
print(gcf,'Fig_Example8_1_05','-painters','-dpng','-r 600')

figure;
for ind = 1:Num
    index = 1:1:info_cano{ind}.iter;
    semilogy(index,info_cano{ind}.Jiter(index)-Jopt,colorName{ind},'linewidth',lineWidth); hold on;
end
ylabel('Suboptimality ($J(K) - J^*$)','Interpreter','latex','FontSize',10);
xlabel('Iterations $t$','Interpreter','latex','FontSize',10);
set(gcf,'Position',[250 150 300 300]);
set(gca,'TickLabelInterpreter','latex')
print(gcf,'Fig_Example8_2_05','-painters','-dpng','-r 600')

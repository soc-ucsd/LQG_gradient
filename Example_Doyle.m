% -------------------------------------------------------------------------
% John Doyle's Example
%-------------------------------------------------------------------------

clc;clear; close all
addpath('./lqg_utils');

nx = 2;    % Number of states
nu = 1;    % Number of inputs;
ny = 1;    % Number of outputs

% dynamics
A = [1 1;0 1];
B = [0;1];
C = [1,0];

q = 5; sigma = 5;
Qc = q*[1;1]*[1 1];
R = 1;
W = sigma*[1;1]*[1 1];
V = 1;

% ---------------------------------------------
% Globally optimal controller
% ---------------------------------------------
S = are(A,B*R^(-1)*B',Qc);
K = lqr(A,B,Qc,R);
P = are(A',C'*V^(-1)*C,W);
L = P*C'*V^(-1);

f = 2 + sqrt(4+q);
d = 2 + sqrt(4+sigma);

% dynamic controller
Ak = A - B*K- L*C;
Bk = L;
Ck = -K;

Ko.Ak = Ak; Ko.Bk = Bk;Ko.Ck = Ck;
% cost
hA    = [A B*Ck;Bk*C Ak];
Y     = lyap(hA',blkdiag(Qc,Ck'*R*Ck));
Jopt  = trace(blkdiag(W,Bk*Bk')*Y);

Num = 1; % number of initial points
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
    K0.Ak = Ak; K0.Bk = Bk;K0.Ck = Ck;

    opts.tol      = 1e-3;
    opts.maxIter  = 2e3;

    % full gradient
    opts.stepsize = 5e-5;
    [K1,J1,info1] = LQG_gd_full(A,B,C,Qc,R,W,V,K0,opts);
    info_full{ind} = info1;
    K_full{ind}    = K1;
    % Hess1 = LQGhessfull(A,B,C,K1,Qc,R,W,V);  % hessian

    % gradient over canonical form
    opts.stepsize = 5e-2;
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
    index = 1:1:info1.iter;
    semilogy(index,info_full{ind}.Jiter(index)-Jopt,colorName{ind},'linewidth',lineWidth); hold on;
end
ylabel('Suboptimality ($J(K) - J^*$)','Interpreter','latex','FontSize',10);
xlabel('Iterations $t$','Interpreter','latex','FontSize',10);
set(gcf,'Position',[250 150 300 300]);
set(gca,'TickLabelInterpreter','latex')

figure;
for ind = 1:Num
    index = 1:1:info1.iter;
    semilogy(index,info_cano{ind}.Jiter(index)-Jopt,colorName{ind},'linewidth',lineWidth); hold on;
end
ylabel('Suboptimality ($J(K) - J^*$)','Interpreter','latex','FontSize',10);
xlabel('Iterations $t$','Interpreter','latex','FontSize',10);
set(gcf,'Position',[250 150 300 300]);
set(gca,'TickLabelInterpreter','latex')


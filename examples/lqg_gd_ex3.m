% -------------------------------------------------------------------------
% The bad Example in the Paper: Theorem 4.4
%-------------------------------------------------------------------------

clc;clear; close all

nx = 2;    % Number of states
nu = 1;    % Number of inputs;
ny = 1;    % Number of outputs

% dynamics
epsilon = 1;
A = 3/2*[-1 0;0 -1-epsilon];
B = [1;1+epsilon];
C = [1,1];

Qc = [4 1;1 4];
R = 1;
V = 1;
W = [4 1+epsilon;1+epsilon 4*(1+epsilon)^2];

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

% ---------------------------------------------
% Gradient descent 
% ---------------------------------------------
p = rand(2,1)-2;%[-1,-2]; % closed-loop poles;
K = place(A,B,p);
L = place(A',C',p); L = L';
Ak = A - B*K- L*C;
Bk = L;
Ck = -K;
K0.Ak = Ak; K0.Bk = Bk;K0.Ck = Ck;

opts.stepsize = 1;
opts.tol      = 1e-6;

% full gradient
[K1,J1,info1] = LQGgd(A,B,C,Qc,R,W,V,K0,opts);
Hess1 = LQGhessfull(A,B,C,K1,Qc,R,W,V);  % hessian

% gradient over canonical form
[K2,J2,info2] = LQGgd_can(A,B,C,Qc,R,W,V,K0,opts);
Hess2 = LQGhessfull(A,B,C,K2,Qc,R,W,V);  % hessian

% another initial point around the globally optimal point
while true
    count = 1;
    K0.Ak = Ak; K0.Bk = Bk;K0.Ck = Ck+0.05*randn(nu,nx);
    Acl  = [A B*K0.Ck; K0.Bk*C K0.Ak];
    if  max(real(eig(Acl))) <0
        break;
    else
        fprintf('initilization %d',count);
        count = count + 1;
    end
end
% K0.Ak = -eye(nx); K0.Bk = zeros(nx,ny);K0.Ck = randn(nu,nx);
% full gradient
[K3,J3,info3] = LQGgd(A,B,C,Qc,R,W,V,K0,opts);
Hess3 = LQGhessfull(A,B,C,K3,Qc,R,W,V);  % hessian

% gradient over canonical form
[K4,J4,info4] = LQGgd_can(A,B,C,Qc,R,W,V,K0,opts);
Hess4 = LQGhessfull(A,B,C,K4,Qc,R,W,V);  % hessian

[J1,J2,J3,J4,Jopt]
([J1,J2,J3,J4]-Jopt)./Jopt

G1 = ss(K1.Ak,K1.Bk,K1.Ck,[]);
G2 = ss(K2.Ak,K2.Bk,K2.Ck,[]);
G3 = ss(K3.Ak,K3.Bk,K3.Ck,[]);
G4 = ss(K4.Ak,K4.Bk,K4.Ck,[]);
Go = ss(Ak,Bk,Ck,[]);

[tf(G1),tf(G2),tf(G3),tf(G4),tf(Go)]

% ------------------------------------------------------------------------
%          Figure
% ------------------------------------------------------------------------

figure;
index = 1:1:info1.iter;
semilogy(index,info1.Jiter(index)-Jopt)
ylabel('Suboptimality ($J(K) - J^*$)','Interpreter','latex','FontSize',10);
xlabel('Iterations $t$','Interpreter','latex','FontSize',10);
set(gcf,'Position',[250 150 400 300]);
set(gca,'TickLabelInterpreter','latex')

figure;
index = 1:1:info2.iter;
semilogy(index,info2.Jiter(index)-Jopt)
ylabel('Suboptimality ($J(K) - J^*$)','Interpreter','latex','FontSize',10);
xlabel('Iterations $t$','Interpreter','latex','FontSize',10);
set(gcf,'Position',[250 150 400 300]);
set(gca,'TickLabelInterpreter','latex')

figure;
index = 1:1:info3.iter;
semilogy(index,info3.Jiter(index)-Jopt)
ylabel('Suboptimality ($J(K) - J^*$)','Interpreter','latex','FontSize',10);
xlabel('Iterations $t$','Interpreter','latex','FontSize',10);
set(gcf,'Position',[250 150 400 300]);
set(gca,'TickLabelInterpreter','latex')

figure;
index = 1:1:info4.iter;
semilogy(index,info4.Jiter(index)-Jopt)
ylabel('Suboptimality ($J(K) - J^*$)','Interpreter','latex','FontSize',10);
xlabel('Iterations $t$','Interpreter','latex','FontSize',10);
set(gcf,'Position',[250 150 400 300]);
set(gca,'TickLabelInterpreter','latex')



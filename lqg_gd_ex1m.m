
%clc;clear;
close all

nx = 2;    % Number of states
nu = 1;    % Number of inputs;
ny = 1;    % Number of outputs

% dynamics
A = rand(2); A = A - 1.1*max(real(eig(A)))*eye(nx);  % make sure it is open-loop stable, thus the initialization works
B = [0;1];
C = [1,0];


% performance weights
Q = 1; R = 1;
Qc = C'*Q*C;
W = eye(nx);V = eye(ny);

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
hA = [A B*Ck;Bk*C Ak];
Y  = lyap(hA',blkdiag(Qc,Ck'*R*Ck));
Jopt  = trace(blkdiag(W,Bk*Bk')*Y);

%[K1,J1,info1] = LQGgd(A,B,C,Qc,R,W,V,Ko);
% ---------------------------------------------
% Gradient descent 
% ---------------------------------------------
K0.Ak = -eye(nx); K0.Bk = zeros(nx,ny);K0.Ck = rand(nu,nx);

[K1,J1,info1] = LQGgd(A,B,C,Qc,R,W,V,K0);
[J1,Jopt,(J1-Jopt)/Jopt]

figure;
index = 1:1:info1.iter;
semilogy(index,info1.Jiter(index)-Jopt)
ylabel('Suboptimality ($J(K) - J^*$)','Interpreter','latex','FontSize',10);
xlabel('Iterations $t$','Interpreter','latex','FontSize',10);
set(gcf,'Position',[250 150 400 300]);
set(gca,'TickLabelInterpreter','latex')


% another initial point
K0.Ak = -eye(nx); K0.Bk = rand(nx,ny);K0.Ck = zeros(nu,nx);

[K2,J2,info2] = LQGgd(A,B,C,Qc,R,W,V,K0);
[J1,J2,Jopt,(J1-Jopt)/Jopt,(J2-Jopt)/Jopt]

figure;
index = 1:1:info2.iter;
semilogy(index,info2.Jiter(index)-Jopt)
ylabel('Suboptimality ($J(K) - J^*$)','Interpreter','latex','FontSize',10);
xlabel('Iterations $t$','Interpreter','latex','FontSize',10);
set(gcf,'Position',[250 150 400 300]);
set(gca,'TickLabelInterpreter','latex')

% another initial point
K0.Ak = Ak; K0.Bk = Bk;K0.Ck = Ck+0.05*randn(nu,nx);

[K3,J3,info3] = LQGgd(A,B,C,Qc,R,W,V,K0);

figure;
index = 1:1:info3.iter;
semilogy(index,info3.Jiter(index)-Jopt)
ylabel('Suboptimality ($J(K) - J^*$)','Interpreter','latex','FontSize',10);
xlabel('Iterations $t$','Interpreter','latex','FontSize',10);
set(gcf,'Position',[250 150 400 300]);
set(gca,'TickLabelInterpreter','latex')



G1 = ss(K1.Ak,K1.Bk,K1.Ck,[]);
G2 = ss(K2.Ak,K2.Bk,K2.Ck,[]);
G3 = ss(K3.Ak,K3.Bk,K3.Ck,[]);

Go = ss(Ak,Bk,Ck,[]);

[tf(G1),tf(G2),tf(G3),tf(Go)]


% ------------------------------------------------------------------------
%          Hessian
% ------------------------------------------------------------------------
[n,m] = size(B);
[p,n] = size(C);

Hess1 = zeros(n^2 + p*n + m*n);
Indx = ones(m+n,p+n);
Indx(1:m,1:p) = zeros(m,p);
Indx = Indx(:);

nonInd = find(Indx > 0);

for i = 1:n^2 + p*n + m*n
    for j = i:n^2 + p*n + m*n   
        Delta1 = zeros((p+n)*(m+n),1);Delta1(nonInd(i)) = 1;
        Delta1 = reshape(Delta1,m+n,p+n); 
        Delta2 = zeros((p+n)*(m+n),1);Delta2(nonInd(j)) = 1;
        Delta2 = reshape(Delta2,m+n,p+n);  
        [Ja, Jb, Jc, H, info] = LQGhess(A,B,C,K1,Qc,R,W,V,Delta1,Delta2);
        Hess1(i,j) = H;
        Hess1(j,i) = H;
    end
end

Hess2 = zeros(n^2 + p*n + m*n);
Indx = ones(m+n,p+n);
Indx(1:m,1:p) = zeros(m,p);
Indx = Indx(:);

nonInd = find(Indx > 0);

for i = 1:n^2 + p*n + m*n
    for j = i:n^2 + p*n + m*n   
        Delta1 = zeros((p+n)*(m+n),1);Delta1(nonInd(i)) = 1;
        Delta1 = reshape(Delta1,m+n,p+n); 
        Delta2 = zeros((p+n)*(m+n),1);Delta2(nonInd(j)) = 1;
        Delta2 = reshape(Delta2,m+n,p+n);  
        [Ja, Jb, Jc, H, info] = LQGhess(A,B,C,K2,Qc,R,W,V,Delta1,Delta2);
        Hess2(i,j) = H;
        Hess2(j,i) = H;
    end
end

Hess3 = zeros(n^2 + p*n + m*n);
Indx = ones(m+n,p+n);
Indx(1:m,1:p) = zeros(m,p);
Indx = Indx(:);

nonInd = find(Indx > 0);

for i = 1:n^2 + p*n + m*n
    for j = i:n^2 + p*n + m*n   
        Delta1 = zeros((p+n)*(m+n),1);Delta1(nonInd(i)) = 1;
        Delta1 = reshape(Delta1,m+n,p+n); 
        Delta2 = zeros((p+n)*(m+n),1);Delta2(nonInd(j)) = 1;
        Delta2 = reshape(Delta2,m+n,p+n);  
        [Ja, Jb, Jc, H, info] = LQGhess(A,B,C,K3,Qc,R,W,V,Delta1,Delta2);
        Hess3(i,j) = H;
        Hess3(j,i) = H;
    end
end


clc;close all
%% Counterexample from Doyle's paper 1978

A = [1 1;0 1];
B = [0;1];
C = [1 0];

% performance weight
q = 0.1;
Q = q*[1;1]*[1 1];
R = 1;

V = 0.1;
w = 100;
W = w*[1;1]*[1 1];

%% LQG design
S = are(A,B*R^(-1)*B',Q);
K1 = R^(-1)*B'*S;
K = lqr(A,B,Q,R);

P = are(A',C'*V^(-1)*C,W);
L = P*C'*V^(-1);

% dynamic controller
Ak = A - B*K- L*C;
Bk = L;
Ck = -K;

%% feasible region

ft = -20:0.2:0;
dt = 0:0.2:10;

Clpole  = zeros(length(ft),length(dt));
LQGcost = zeros(length(ft),length(dt));
for i = 1:length(ft)
    for j = 1:length(dt)
        f = ft(i);
        d = dt(j);
        
        Bk = f*[1;1];
        Ck = d*[1 1];
        Ak = A - B*Ck + Bk*C;
        Acl = [A B*Ck;Bk*C Ak];
        
        Clpole(i,j) = max(real(eig(Acl)));
        
        if Clpole(i,j) <  -1e-6
            hA = Acl;
            Y  = lyap(hA',blkdiag(Q,Ck'*R*Ck));
            if min(eig(Y)) < 0
                break;
            end
            LQGcost(i,j) = trace(blkdiag(W,Bk*V*Bk')*Y);
        else
            LQGcost(i,j) = nan;
        end
        
    end
end

figure;
[X,Y] = meshgrid(dt,ft);
h = surface(X,Y,Clpole,'FaceAlpha',0.5);
%h.EdgeColor = 'none';
colorbar
%view(3)
view(90,0)


%% Cost function
figure;
h = surface(X,Y,LQGcost,'FaceAlpha',0.5);
zlim([0 5000])
colorbar
%view(3)
view(90,0)
[i,j] = find(LQGcost == min(min(LQGcost)));
f = ft(i)
d = dt(j)



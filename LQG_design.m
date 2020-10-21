
clc;clear;%close all

nx = 2;    %Number of states
ny = 1;    %Number of outputs

% dynamics
%A = rand(nx);%
A = [1 2;3 1];% - 5*eye(2);
B = [0;1];
C = [1,0];
sys = ss(A,B,C,[],[]);


%% From Doyle's paper 
A = [1 1;0 1];
B = [0;1];
C = [1 0];

% performance weight
q = 100;
Q = q*[1;1]*[1 1];
R = 1;

V = 1;
w = 100;
W = w*[1;1]*[1 1];

Qc = Q;

% % performance weights
% Q = 1;
% R = 1;
% Qc = C'*Q*C;

%% continous LQR

S = are(A,B*R^(-1)*B',Qc);
K1 = R^(-1)*B'*S;
K = lqr(A,B,Qc,R);

%W = eye(nx);V = eye(ny);
P = are(A',C'*V^(-1)*C,W);
L = P*C'*V^(-1);

% dynamic controller
Ak = A - B*K- L*C;
Bk = L;
Ck = -K;


% cost
hA = [A B*Ck;Bk*C Ak];
Y  = lyap(hA',blkdiag(Qc,Ck'*R*Ck));
Jopt  = trace(blkdiag(W,Bk*Bk')*Y);

% T = rand(nx,nx);
% Ak = T*Ak*T^(-1);
% Bk = T*Bk;
% Ck = Ck*T^(-1);

Ks.Ak = Ak; Ks.Bk = Bk; Ks.Ck = Ck;
flag =  1; % continous time systems
[Grad_A,Grad_B,Grad_C,J,info] = LQGgrad(A,B,C,Ks,Qc,R,W,V,flag);


%% Similarity transformation 
Q1 = ctrb(Ak,Bk);
tmp = charpoly(Ak);
At = [0 1;
      -tmp(3) -tmp(2)];
Q2 = ctrb(At,[0;1]);

T = Q2*Q1^(-1);

Bt = T*Bk; Ct = Ck*T^(-1);

%% Gradient descent 

theta = [At(2,1:end), Ct];
theta(1) = theta(1) + (2*rand(1)-1)*0.1;
theta(3) = theta(3)+(2*rand(1)-1);    % this initial controller might be unstable
theta(4) = theta(4)+(2*rand(1)-1);


Ak = [0 1; theta(1) theta(2)];
Bk = [0;1];
Ck = [theta(3) theta(4)];
Kt.Ak = Ak; Kt.Bk = Bk; Kt.Ck = Ck;

maxIter = 1e4;
thetaIter = zeros(4,maxIter);
gradIter  = zeros(4,maxIter);  
Jcost     = zeros(maxIter,1);
gradNorm  = zeros(maxIter,1);


stepsize  = 5e-4;

thetaOld = theta;

for k = 1:maxIter
    
    [Grad_A,Grad_B,Grad_C,Jt,Info] = LQGgrad(A,B,C,Kt,Qc,R,W,V,flag);
    
    gradNorm(k) = norm([Grad_A(:);Grad_B(:);Grad_C(:)],2)^2;
    
    Jcost(k) = Jt;
    
  %  stepsize  = 0.1;
    Grad = [Grad_A(2,1:end),Grad_C];
    thetaNew = thetaOld - stepsize*Grad;
    Ak = [0 1; thetaNew(1) thetaNew(2)];
    Bk = [0;1];
    Ck = [thetaNew(3) thetaNew(4)];
    
    while true     
        hA = [A B*Ck;Bk*C Ak];
        %stepsize = 0.1;
        if (max(real(eig(hA)))>=0)
            stepsize = stepsize/2;
            thetaNew = thetaOld - stepsize*Grad;
            Ak = [0 1; thetaNew(1) thetaNew(2)];
            Bk = [0;1];
            Ck = [thetaNew(3) thetaNew(4)];
            if stepsize < 1e-6
                warning('step size is too small');
                break;
            end
        else
            break;
        end
    end
    Kt.Ak = Ak; Kt.Bk = Bk; Kt.Ck = Ck;
    
    thetaIter(:,k) = thetaNew(:);
    gradIter(:,k)  = Grad(:);
    thetaOld       = thetaNew;

    
    if mod(k,50) == 0 || k == 1
        fprintf('Iter %4d     Cost %5.4f   Err %5.3f  Gradient %5.4f Full_grad %5.4f  Step Size %5.4f\n',k, Jt, (Jt - Jopt)/Jopt, norm(Grad),gradNorm(k), stepsize);
    end
    
    if norm(Grad) < 1e-4
        break;
    end
end



figure;
index = 10:10:k;
semilogy(index,Jcost(index)-Jopt)

%%
figure
plot(gradNorm(1:k),Jcost(1:k)- Jopt);
xlim([0,100])
ylabel('J - J^*');
xlabel('Norm of gradient')


Ko = ss(Ks.Ak,Ks.Bk,Ks.Ck,[]);
K1 = ss(Kt.Ak,Kt.Bk,Kt.Ck,[]);
[norm(Ko),norm(K1),norm(Ko-K1)]

%% No structures

% 
% Ak = [0 1; theta(1) theta(2)];
% Bk = [0;1];
% Ck = [theta(3) theta(4)];
% Kt.Ak = Ak; Kt.Bk = Bk; Kt.Ck = Ck;
% 
% thetaIter = zeros(4,maxIter);
% gradIter  = zeros(4,maxIter);  
% Jcost     = zeros(maxIter,1);
% 
% stepsize  = 0.1;
% 
% AkOld = Ak;
% BkOld = Bk;
% CkOld = Ck;
% %% no controller structure
% 
% fprintf('===================================\n')
% 
% for k = 1:maxIter
%     
%     [Grad_A,Grad_B,Grad_C,Jt,Info] = LQGgrad(A,B,C,Kt,Qc,R,W,V,flag);
%     
%     Jcost(k) = Jt;
%     
%     Grad = [Grad_A(2,1:end),Grad_C];
%  %   stepsize  = 0.1;
%     Ak = AkOld - stepsize * Grad_A;
%     Bk = BkOld - stepsize * Grad_B;
%     Ck = CkOld - stepsize * Grad_C;
%     
%     while true     
%         hA = [A B*Ck;Bk*C Ak];
%         if (max(real(eig(hA)))>=0)
%             stepsize = stepsize/2;
%             Ak = AkOld - stepsize * Grad_A;
%             Bk = BkOld - stepsize * Grad_B;
%             Ck = CkOld - stepsize * Grad_C;
%             if stepsize < 1e-12
%                 warning('step size is too small');
%                 break;
%             end
%         else
%             break;
%         end
%     end
%     Kt.Ak = Ak; Kt.Bk = Bk; Kt.Ck = Ck;
%     
%     gradIter(:,k)  = Grad(:);
%     AkOld       = Ak;
%     BkOld       = Bk;
%     CkOld       = Ck;
% 
%     
%     if mod(k,50) == 0 || k == 1
%         fprintf('Iter %4d     Cost %5.4f   Err %5.3f  Gradient %5.4f \n',k, Jt, (Jt - J)/J, norm(Grad));
%     end
%     
%     if norm(Grad) < 5e-2
%         break;
%     end
% end
% 
% figure;
% index = 10:10:k;
% loglog(index,Jcost(index))
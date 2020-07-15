
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
A = rand(2);%[1 1;0 1];
B = rand(2,1); %[0;1];
C = rand(1,2); %[1 0];

% performance weight
q = 1;
Q = q*[1;1]*[1 1];
R = 1;

V = 1;
w = 1;
W = w*[1;1]*[1 1];

Qc = Q;



%%
Num = 2e4;
JcostNew     = zeros(Num,1);
gradNormNew  = zeros(Num,1);


S = are(A,B*R^(-1)*B',Qc);
K1 = R^(-1)*B'*S;
K = lqr(A,B,Qc,R);

%W = eye(nx);V = eye(ny);
P = are(A',C'*V^(-1)*C,W);
L = P*C'*V^(-1);

% dynamic controller
Akopt = A - B*K- L*C;
Bkopt = L;
Ckopt = -K;

% cost
hA = [A B*Ckopt;Bkopt*C Akopt];
Y  = lyap(hA',blkdiag(Qc,Ckopt'*R*Ckopt));
Jopt  = trace(blkdiag(W,Bkopt*Bkopt')*Y);


flag =1 ;
   index = 1;
while index <= Num
   Ak = Akopt + 0.0005*randn(2);
   Bk = Bkopt + 0.0005*randn(2,1);
   Ck = Ckopt + 0.0005*randn(1,2);
   
    hA = [A B*Ck;Bk*C Ak];
    
    Kt.Ak = Ak;Kt.Bk = Bk; Kt.Ck = Ck; 
 
    if max(real(eig(hA))) < -1e-6
        [Grad_A,Grad_B,Grad_C,Jt,Info] = LQGgrad(A,B,C,Kt,Qc,R,W,V,flag);

        gradNormNew(index) = norm([Grad_A(:);Grad_B(:);Grad_C(:)],2)^2;
        
        JcostNew(index) = Jt;
        
        
        if mod(index,100) == 0
            fprintf('index %4d grad %f cost %f\n', index, gradNormNew(index), JcostNew(index));
        end
        
         index = index + 1;
       
    end  
end





%%
figure
scatter(gradNormNew,JcostNew- Jopt,'.');
xlim([0,1])
ylabel('J - J^*');
xlabel('Norm of gradient')


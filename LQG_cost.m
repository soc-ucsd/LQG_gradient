
%% LQG cost function 
close all

A = [1 2;3 1]/10;
B = [0;1];
C = [1,0];
Q = 1;
R = 1;

theta = zeros(4,1);  % controller

Num = 50;

J = zeros(Num,Num);

a = linspace(-5,5,Num);
b = linspace(-5,5,Num);
theta(2) = 0.4;
for i = 1:Num
    
    theta(1) = b(i);
    for j = 1:Num

    
    theta(3) = a(j);
    
    Ak = [0 1;
          theta(1), theta(2)];
    Bk = [0;1];
    Ck = [theta(3), theta(4)];
    
    
    hA = [A B*Ck;Bk*C Ak];
    if max(abs(eig(hA))) < 0.99
        Y = dlyap(hA',blkdiag(C'*Q*C,Ck'*R*Ck));
        J(i,j) = trace(blkdiag(eye(2),Bk*Bk')*Y);
    else
        J(i,j) = inf;
    end
    end
end
figure
spy(J<inf)

figure
surface(J)
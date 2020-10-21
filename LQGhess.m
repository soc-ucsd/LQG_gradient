function [Ja, Jb, Jc, H, info] = LQGhess(A,B,C,K,Q,R,W,V,Delta1,Delta2)
% Compute the hessian of LQG cost function at K
% and direction Delta1, Delta2

    % 
    flag = 1;
    nx = size(A,1);
    p = size(C,1);
    m  = size(B,2);
 
    % controller paramters
    Ak = K.Ak;
    Bk = K.Bk;
    Ck = K.Ck;

    hA = [A B*Ck; Bk*C Ak];   % closed-loop system matrix
    
    %
    D1 = Delta1 + Delta2;
    D2 = Delta1 - Delta2;
    if flag == 1      % continous time
        if max(real(eig(hA))) >= 0
            error('The closed-loop system is not stable');
        end
        Y  = lyap(hA',blkdiag(Q,Ck'*R*Ck));
        X  = lyap(hA,blkdiag(W,Bk*V*Bk'));
        
        % Gradient information
        Y1 = Y(1:nx,1:nx);
        Y2 = Y(1:nx,nx+1:end);
        Y3 = Y(nx+1:end,nx+1:end);

        X1 = X(1:nx,1:nx);
        X2 = X(1:nx,nx+1:end);
        X3 = X(nx+1:end,nx+1:end);

        Ja = 2*(Y2'*X2 + Y3*X3);
        Jb = 2*(Y3*Bk*V + Y3*X2'*C' + Y2'*X1*C');
        Jc = 2*(R*Ck*X3 + B'*Y1*X2 + B'*Y2*X3);
        
        % for debug
        hB = blkdiag(B,eye(nx));
        hC = blkdiag(C,eye(nx));
        info.grad = 2*(hB'*Y + [zeros(m,nx) R*Ck; zeros(nx,nx), zeros(nx,nx)])*....
            (hC*X + [zeros(p,nx) V*Bk'; zeros(nx,nx), zeros(nx,nx)])';

        % Hessian information - D1
        D1Ak = D1(p+1:p+nx,m+1:m+nx);
        D1Bk = D1(p+1:p+nx,1:m);
        D1Ck = D1(1:p,m+1:m+nx);
        tmp = Bk*V*D1Bk';
        M1 = hB*D1*hC*X + X*hC'*D1'*hB' + blkdiag(zeros(nx,nx),tmp+tmp');
        hX = lyap(hA,M1);
        HessD1 = 2*trace(2*hB*D1*hC*hX*Y + 2*blkdiag(zeros(nx,nx),Ck'*R*D1Ck)*hX ...
                                        + blkdiag(zeros(nx,nx),D1Bk*V*D1Bk')*Y ...
                                        + blkdiag(zeros(nx,nx),D1Ck'*R*D1Ck)*X);
        
        % Hessian information - D2                            
        D2Ak = D2(p+1:p+nx,m+1:m+nx);
        D2Bk = D2(p+1:p+nx,1:m);
        D2Ck = D2(1:p,m+1:m+nx);
        tmp = Bk*V*D2Bk';
        M1 = hB*D2*hC*X + X*hC'*D2'*hB' + blkdiag(zeros(nx,nx),tmp+tmp');
        hX = lyap(hA,M1);
        HessD2 = 2*trace(2*hB*D2*hC*hX*Y + 2*blkdiag(zeros(nx,nx),Ck'*R*D2Ck)*hX ...
                                        + blkdiag(zeros(nx,nx),D2Bk*V*D2Bk')*Y ...
                                        + blkdiag(zeros(nx,nx),D2Ck'*R*D2Ck)*X);
        H = 1/4*(HessD1 - HessD2);
        
    elseif flag == 0    % discrete time
           % to do
    end

end


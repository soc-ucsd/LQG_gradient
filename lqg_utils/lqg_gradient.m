function [Ja, Jb, Jc,J,info] = lqg_gradient(A,B,C,K,Q,R,W,V,flag)
% Compute the gradient of LQG cost function

    nx = size(A,1);
    p = size(C,1);
    m  = size(B,2);
    % controller paramters
    Ak = K.Ak;
    Bk = K.Bk;
    Ck = K.Ck;

    hA = [A B*Ck; Bk*C Ak];   % closed-loop system matrix
    if flag == 1      % continous time
        if max(real(eig(hA))) >= 0
            error('The closed-loop system is not stable');
        end
        Y  = lyap(hA',blkdiag(Q,Ck'*R*Ck));
        X  = lyap(hA,blkdiag(W,Bk*V*Bk'));
        
        J = trace(blkdiag(W,Bk*V*Bk')*Y);

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

    elseif flag == 0    % discrete time
           % to do
    end

end


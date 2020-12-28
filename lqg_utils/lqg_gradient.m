function [Ja, Jb, Jc,J,info] = lqg_gradient(A,B,C,K,Q,R,W,V,flag)
% Compute the gradient of LQG cost function
% Flag:   1 --> continuous time
%         0 --> discrete time
%
%  Ja, Jb, Jc: partial gradient over Ak, Bk, Ck
%  J         : LQG cost value

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
        if max(abs(eig(hA))) >= 1
            error('The closed-loop system is not stable');
        end
        
        Y  = dlyap(hA',blkdiag(Q,Ck'*R*Ck));
        X  = dlyap(hA,blkdiag(W,Bk*V*Bk'));
        
        J = trace(blkdiag(W,Bk*V*Bk')*Y);

        % partition
        %  Y = [Y11  Y12
        %       Y12' Y22];
        Y1 = Y(1:nx,1:nx);          % Y11
        Y2 = Y(1:nx,nx+1:end);      % Y12
        Y3 = Y(nx+1:end,nx+1:end);  % Y22

        X1 = X(1:nx,1:nx);
        X2 = X(1:nx,nx+1:end);
        X3 = X(nx+1:end,nx+1:end);

        Ja = 2*(Y2'*(A*X2+B*Ck*X3) + Y3*Ak*X3 + Y3*Bk*C*X2);
        Jb = 2*(Y2'*(A*X1 + B*Ck*X2')*C' + Y3*Ak*X2'*C' + Y3*Bk*(C*X1*C'+V));
        Jc = 2*(B'*Y2*(Ak*X3 + Bk*C*X2) + B'*Y1*A*X2 + (B'*Y1*B+R)*Ck*X3);
        
        % for debug
        hB = blkdiag(B,eye(nx));
        hC = blkdiag(C,eye(nx));
        info.grad = 2*([zeros(m,nx) R*Ck; zeros(nx,nx), zeros(nx,nx)]*X*blkdiag(zeros(nx,m),eye(nx)) + ...
            blkdiag(zeros(p,nx),eye(nx))*Y*[zeros(nx,p) zeros(nx,nx); Bk*V, zeros(nx,nx)] + hB'*Y*hA*X*hC');
    end

end


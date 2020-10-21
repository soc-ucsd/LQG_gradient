function [Kopt,Jopt,info] = LQGgd(A,B,C,Q,R,W,V,K0)
% Gradient descent algorithm for LQG problem
%
%     K_{t+1} = K_t - lambda * Grad(K_t)
%  where we take a full gradient, and do not account for canonical form
%        Step size is chosen via the Armijo Rule
%  
%    Dynamics: A, B, C
%    Weights:  Q, R
%    Noise:    W,V
%    Initial points: K0

flag = 1; % continuous time systems
[n,m] = size(B);
[p,~] = size(C);

% ------------------------------------------------------------------------
% Setup parameters  
% ------------------------------------------------------------------------
alpha   = 0.01;     % backtrapping line search 
beta    = 0.5;
tol     = 1e-6;    % tolerance of norm of gradient direction
MaxIter = 1e4;      % maximum number of gradient steps
Disp    = 50;

Jcost   = zeros(MaxIter,0);
 
% ------------------------------------------------------------------------
% Gradient descent iteration  
% ------------------------------------------------------------------------
K    = K0; 
Acl  = [A B*K.Ck; K.Bk*C K.Ak];
if  max(real(eig(Acl))) >=0
    error('The initial point is not a stabilizing controller.\n')
else
    Y = lyap(Acl,blkdiag(Q,K.Ck'*R*K.Ck));
    J = trace(blkdiag(W,K.Bk*V*K.Bk')*Y);   % initial cost value
end

fprintf('Iter       ngradK      LQG cost   Step_size\n') 
for Iter = 1:MaxIter 
    
    Jcost(Iter) = J;     % the LQG cost in the current step; 
    % --------------------------------------------------------------------
    %    compute the gradient projection
    % --------------------------------------------------------------------
    [Grad_A,Grad_B,Grad_C,~,~] = LQGgrad(A,B,C,K,Q,R,W,V,flag);
    gradK = [zeros(m,p) Grad_C;
            Grad_B,    Grad_A];  % put it into one matrix    
    ngradK = norm(gradK,'fro');

    % stop the algorithm if the norm of gradient is small enough        
    if ngradK < tol
        break;
    end
    
    % --------------------------------------------------------------------
    %    Update according to Armijo rule: 
    % -------------------------------------------------------------------     
    StepSize = 1;
    Kt.Ak    = K.Ak - StepSize*Grad_A;
    Kt.Bk    = K.Bk - StepSize*Grad_B;
    Kt.Ck    = K.Ck - StepSize*Grad_C;
    
    Acl      = [A B*Kt.Ck;Kt.Bk*C Kt.Ak];
    mEigAcl  = max(real(eig(Acl)));
    Y        = lyap(Acl',blkdiag(Q,Kt.Ck'*R*Kt.Ck));
    Jtemp    = trace(blkdiag(W,Kt.Bk*Kt.Bk')*Y);

    
    % Backtracking line search
    while mEigAcl >= 0 || J - Jtemp < StepSize*alpha*trace(gradK'*gradK)
        StepSize = beta*StepSize;
        if StepSize < 1.e-19
            warning('Gradient method gets stuck with very small step size!');
            break;
        end
        Kt.Ak    = K.Ak - StepSize*Grad_A;
        Kt.Bk    = K.Bk - StepSize*Grad_B;
        Kt.Ck    = K.Ck - StepSize*Grad_C;
        Acl      = [A B*Kt.Ck;Kt.Bk*C Kt.Ak];
        mEigAcl  = max(real(eig(Acl)));
        Y        = lyap(Acl',blkdiag(Q,Kt.Ck'*R*Kt.Ck));
        Jtemp    = trace(blkdiag(W,Kt.Bk*Kt.Bk')*Y);
    end

    if mod(Iter,Disp) == 0 || Iter == 1
        fprintf('%4d     %6.4E    %6.4E   %6.4E \n',Iter, ngradK, J, StepSize);
    end
    
    % update the current step K
    K = Kt;
    J = Jtemp;

    % stop the algorithm if the norm of gradient is small enough        
%     if ngradK < tol
%         break;
%     end
end

% -----------------------------------------------------------------------
% Output information
% -----------------------------------------------------------------------
Kopt = K;
Jopt = J;

[Grad_A,Grad_B,Grad_C,J1,~] = LQGgrad(A,B,C,Kopt,Q,R,W,V,flag);
info.Jopt    = J1;
info.Jiter   = Jcost(1:Iter);
info.grad.Ak = Grad_A;
info.grad.Bk = Grad_B;
info.grad.Ck = Grad_C;
info.iter    = Iter;
end


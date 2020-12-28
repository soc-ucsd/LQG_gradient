function [Kopt,Jopt,info] = LQG_gd_full(A,B,C,Q,R,W,V,K0,opts)
%
% LQG: Gradient descent algorithm with a full gradient update
%
%           K_{t+1} = K_t - s_t * Grad(K_t)
%
%    where we take a partial gradient in a canonical form
%    Step size s_t is chosen via the Armijo Rule
%    Works for MIMO systems
%
% Inputs:
%    Dynamics:       A, B, C
%    Weights:        Q, R
%    Noise level:    W,V
%    Initial point:  K0, which contains K0.Ak, K0.Bk, K0.Ck
%
%    User options: opts
%                    opts.tol        stopping tolorance
%                    opts.stepsize   step size for line search
%                    opts.maxIter    maximum iterations
% Outputs:
%    Kopt:  optimal controller
%    Jopt:  optimal LQG cost
%    info:  some output information

% Authors: Yand Zheng, Yujie Tang, Na Li
% Paper:   Analysis of the Optimization Landscape of Linear Quadratic Gaussian Control


flag = 1; % continuous time systems
[n,m] = size(B);
[p,~] = size(C);

% ------------------------------------------------------------------------
% Setup default parameters  
% ------------------------------------------------------------------------
initsize = 1;
alpha    = 0.2;     % backtrapping line search 
beta     = 0.5;
tol      = 1e-8;    % tolerance of norm of gradient direction
MaxIter  = 1e4;      % maximum number of gradient steps
Disp     = 100;

myline1 = [repmat('=',1,48),'\n'];
myline2 = [repmat('-',1,48),'\n'];
header  = ' iter  |   ngradK    |   LQG cost  | step_size \n';


%============================================
% Setup
%============================================
% Set user options
if(nargin > 8)
    initsize = opts.stepsize;
    tol      = opts.tol;
    MaxIter  = opts.maxIter;
    Disp     = opts.Disp;
end

% ------------------------------------------------------------------------
% Initial stabilization 
% ------------------------------------------------------------------------
Jcost = zeros(MaxIter,0); 
K     = K0; 
Acl   = [A B*K.Ck; K.Bk*C K.Ak];
if  max(real(eig(Acl))) >=0
    error('The initial point is not a stabilizing controller.\n')
else
    Y = lyap(Acl,blkdiag(Q,K.Ck'*R*K.Ck));
    J = trace(blkdiag(W,K.Bk*V*K.Bk')*Y);   % initial cost value
end

% ------------------------------------------------------------------------
% Gradient descent iteration  
% ------------------------------------------------------------------------
fprintf(myline1);
fprintf('Gradient descent for LQG problem\n');
fprintf('System dimensions: n = %d, m = %d, p = %d\n',n,m,p);
fprintf('Maximum iter.    : %6.2E\n',MaxIter);
fprintf('Stopping tol.    : %6.2E\n',tol);
fprintf(myline2);
fprintf(header); 

for Iter = 1:MaxIter 
    
    Jcost(Iter) = J;     % the LQG cost in the current step; 
    % --------------------------------------------------------------------
    %    compute the gradient projection
    % --------------------------------------------------------------------
    [Grad_A,Grad_B,Grad_C,~,~] = lqg_gradient(A,B,C,K,Q,R,W,V,flag);
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
    StepSize = initsize;
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
        fprintf('%4d     %6.4E    %6.4E    %6.3E \n',Iter, ngradK, J, StepSize);
    end
    
    % update the current step K
    K = Kt;
    J = Jtemp;
end

% -----------------------------------------------------------------------
% Output information
% -----------------------------------------------------------------------
fprintf(myline2);
fprintf('Final LQG cost: %6.3E\n',J);
fprintf('Grad norm     : %6.3E\n',ngradK);

Kopt = K;
Jopt = J;

[Grad_A,Grad_B,Grad_C,J1,~] = lqg_gradient(A,B,C,Kopt,Q,R,W,V,flag);
info.Jopt    = J1;
info.Jiter   = Jcost(1:Iter);
info.grad.Ak = Grad_A;
info.grad.Bk = Grad_B;
info.grad.Ck = Grad_C;
info.iter    = Iter;
end


function [Kopt,Jopt,info] = LQG_gd_cano(A,B,C,Q,R,W,V,K0,userOpts)
%
% LQG: Gradient descent algorithm over a controllable canonical form
%
%           K_{t+1} = K_t - s_t * Grad(K_t)
%
%    where we take a partial gradient in a canonical form
%    Step size s_t is chosen via the Armijo Rule
%    Only works for SISO systems now
%
% Inputs:
%    Dynamics:       A, B, C
%    Weights:        Q, R
%    Noise level:    W,V
%    Initial point:  K0, which contains K0.Ak, K0.Bk, K0.Ck
%
%    User options: opts
%                    opts.opts.tol        stopping opts.tolorance
%                    opts.stepsize   step size for line search
%                    opts.opts.maxIter    maximum iterations
% Outputs:
%    Kopt:  optimal controller
%    Jopt:  optimal LQG cost
%    info:  some output information

% Authors: Yand Zheng, Yujie Tang, Na Li
% Paper:   Analysis of the Optimization Landscape of Linear Quadratic Gaussian Control

% System dimensions
flag  = 1;         % continuous time systems for now
[n,m] = size(B);
[p,~] = size(C);

%------------------------------------------------------------------------
% Setup default parameters  
%------------------------------------------------------------------------
opts.stepsize = 1;
opts.alpha    = 0.2;     % backtrapping line search 
opts.beta     = 0.5;
opts.tol      = 1e-8;    % opts.tolerance of norm of gradient direction
opts.maxIter  = 1e3;     % maximum number of gradient steps
opts.Disp     = 100;

myline1 = [repmat('=',1,64),'\n'];
myline2 = [repmat('-',1,64),'\n'];
header  = ' iter  |   ngradK    |   par_ngradK  |   LQG cost   |  step_size \n';

%------------------------------------------------------------------------
% Setup
%------------------------------------------------------------------------
% Set user options
if(nargin > 8)
    fnames = fieldnames(userOpts);
    for i=1:length(fnames)
        if isfield(opts,fnames{i})
            opts.(fnames{i}) = userOpts.(fnames{i});
        else
            warning('Option ''%s'' is unknown and will be ignored.',fnames{i})
        end
    end
end

% ------------------------------------------------------------------------
% Initial stabilization 
% ------------------------------------------------------------------------
Jcost = zeros(opts.maxIter,0); 
K     = K0; 
Acl   = [A      B*K.Ck; 
         K.Bk*C K.Ak];
if  max(real(eig(Acl))) >=0
    error('The initial point is not a stabilizing controller.\n')
else
    Y = lyap(Acl,blkdiag(Q,K.Ck'*R*K.Ck));
    J = trace(blkdiag(W,K.Bk*V*K.Bk')*Y);   % initial cost value
end

% ------------------------------------------------------------------------
% Controllability canonical form  
% ------------------------------------------------------------------------
Q1  = ctrb(K.Ak,K.Bk);
tmp = charpoly(K.Ak);
Ak  = [zeros(n-1,1), eye(n-1);
       -tmp(end:-1:2)];
Bk  = zeros(n,1); Bk(end) = 1;
Q2  = ctrb(Ak,Bk);
T   = Q2*Q1^(-1);
Bk  = T*K.Bk; Ck = K.Ck*T^(-1);

K.Ak = Ak; K.Bk = Bk; K.Ck = Ck;

% ------------------------------------------------------------------------
% Gradient descent algorithm  
% ------------------------------------------------------------------------
fprintf(myline1);
fprintf('Gradient descent for LQG problem\n');
fprintf('System dimensions: n = %d, m = %d, p = %d\n',n,m,p);
fprintf('Maximum iter.    : %6.2E\n',opts.maxIter);
fprintf('Stopping opts.tol.    : %6.2E\n',opts.tol);
fprintf(myline2);
fprintf(header); 

for Iter = 1:opts.maxIter 
    
    Jcost(Iter) = J;     % the LQG cost in the current step; 
    % --------------------------------------------------------------------
    %    compute the gradient projection
    % --------------------------------------------------------------------
    [Grad_A,Grad_B,Grad_C,~,~] = lqg_gradient(A,B,C,K,Q,R,W,V,flag);
    gradK = [zeros(m,p) Grad_C;
            Grad_B,    Grad_A];  % put it into one matrix     
    ngradK = norm(gradK,'fro');
   
    % partial gradient
    tmp = zeros(n,n); tmp(end,:) = Grad_A(end,:);
    par_gradK = [zeros(m,p) Grad_C;
                 zeros(n,p),tmp];
    npar_gradK = norm(par_gradK,'fro');

    % stop the algorithm if the norm of gradient is small enough        
    if npar_gradK < opts.tol
        break;
    end
    
    % --------------------------------------------------------------------
    %    Update according to Armijo rule: 
    % -------------------------------------------------------------------     
    StepSize = opts.stepsize;
    parA = zeros(n,n); parA(end,:) = Grad_A(end,:);
    Kt.Ak    = K.Ak - StepSize*parA;
    Kt.Bk    = K.Bk;
    Kt.Ck    = K.Ck - StepSize*Grad_C;
    
    Acl      = [A B*Kt.Ck;Kt.Bk*C Kt.Ak];
    mEigAcl  = max(real(eig(Acl)));                  % stable or not
    Y        = lyap(Acl',blkdiag(Q,Kt.Ck'*R*Kt.Ck));
    Jtemp    = trace(blkdiag(W,Kt.Bk*Kt.Bk')*Y);     % LQG cost

    % Backtracking line search
    while mEigAcl >= 0 || J - Jtemp < StepSize*opts.alpha*trace(par_gradK'*par_gradK)
        StepSize = opts.beta*StepSize;
        if StepSize < 1.e-19
            warning('Gradient method gets stuck with very small step size!');
            break;
        end
        parA = zeros(n,n); parA(end,:) = Grad_A(end,:);
        Kt.Ak    = K.Ak - StepSize*parA;
        Kt.Bk    = K.Bk;
        Kt.Ck    = K.Ck - StepSize*Grad_C;
        Acl      = [A B*Kt.Ck;Kt.Bk*C Kt.Ak];
        mEigAcl  = max(real(eig(Acl)));                    % stable or not
        Y        = lyap(Acl',blkdiag(Q,Kt.Ck'*R*Kt.Ck));   
        Jtemp    = trace(blkdiag(W,Kt.Bk*Kt.Bk')*Y);       % LQG cost
    end

    if mod(Iter,opts.Disp) == 0 || Iter == 1
        fprintf('%4d     %6.4E      %6.4E     %6.4E    %6.4E \n',Iter, ngradK, npar_gradK, J, StepSize);
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
fprintf('Grad norm     : %6.3E\n',npar_gradK);

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


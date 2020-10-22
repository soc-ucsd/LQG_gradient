function [Kopt,Jopt,info] = LQGgd_can(A,B,C,Q,R,W,V,K0,opts)
% Gradient descent algorithm for LQG problem
%
%     K_{t+1} = K_t - lambda * Grad(K_t)
%  where we take a partial gradient in a canonical form
%  Step size is chosen via the Armijo Rule
%  Only works for SISO systems now
%  
%    Dynamics: A, B, C
%    Weights:  Q, R
%    Noise:    W,V
%    Initial points: K0

flag = 1; % continuous time systems
[n,m] = size(B);
[p,~] = size(C);

% ------------------------------------------------------------------------
% Setup default parameters  
% ------------------------------------------------------------------------
initsize = 1;
alpha    = 1e-2;     % backtrapping line search 
beta     = 0.5;
tol      = 1e-4;    % tolerance of norm of gradient direction
MaxIter  = 1e4;      % maximum number of gradient steps
Disp     = 50;

%============================================
% Setup
%============================================
% Set user options
if(nargin > 8)
    initsize = opts.stepsize;
end

%
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


% ------------------------------------------------------------------------
% Controllability canonical form  
% ------------------------------------------------------------------------
Q1  = ctrb(K.Ak,K.Bk);
tmp = charpoly(K.Ak);
Ak  = [zeros(n-1,1), eye(n-1);
       -tmp(end:-1:2)];
Bk  = zeros(n,1); Bk(end) = 1;
Q2 = ctrb(Ak,Bk);
T = Q2*Q1^(-1);
Bk = T*K.Bk; Ck = K.Ck*T^(-1);

K.Ak = Ak; K.Bk = Bk; K.Ck = Ck;

fprintf('Iter       ngradK     part_grad     LQG cost   Step_size\n') 
for Iter = 1:MaxIter 
    
    Jcost(Iter) = J;     % the LQG cost in the current step; 
    % --------------------------------------------------------------------
    %    compute the gradient projection
    % --------------------------------------------------------------------
    [Grad_A,Grad_B,Grad_C,~,~] = LQGgrad(A,B,C,K,Q,R,W,V,flag);
    gradK = [zeros(m,p) Grad_C;
            Grad_B,    Grad_A];  % put it into one matrix     
    ngradK = norm(gradK,'fro');
   
    % partial gradient
    tmp = zeros(n,n); tmp(end,:) = Grad_A(end,:);
    par_gradK = [zeros(m,p) Grad_C;
                 zeros(n,p),tmp];
    npar_gradK = norm(par_gradK,'fro');

    % stop the algorithm if the norm of gradient is small enough        
    if npar_gradK < tol
        break;
    end
    
    % --------------------------------------------------------------------
    %    Update according to Armijo rule: 
    % -------------------------------------------------------------------     
    StepSize = initsize;
    parA = zeros(n,n); parA(end,:) = Grad_A(end,:);
    Kt.Ak    = K.Ak - StepSize*parA;
    Kt.Bk    = K.Bk;% - StepSize*Grad_B;
    Kt.Ck    = K.Ck - StepSize*Grad_C;
    
    Acl      = [A B*Kt.Ck;Kt.Bk*C Kt.Ak];
    mEigAcl  = max(real(eig(Acl)));
    Y        = lyap(Acl',blkdiag(Q,Kt.Ck'*R*Kt.Ck));
    Jtemp    = trace(blkdiag(W,Kt.Bk*Kt.Bk')*Y);

    
    % Backtracking line search
    while mEigAcl >= 0 || J - Jtemp < StepSize*alpha*trace(par_gradK'*par_gradK)
        StepSize = beta*StepSize;
        if StepSize < 1.e-19
            warning('Gradient method gets stuck with very small step size!');
            break;
        end
        parA = zeros(n,n); parA(end,:) = Grad_A(end,:);
        Kt.Ak    = K.Ak - StepSize*parA;
        Kt.Bk    = K.Bk;% - StepSize*Grad_B;
        Kt.Ck    = K.Ck - StepSize*Grad_C;
        Acl      = [A B*Kt.Ck;Kt.Bk*C Kt.Ak];
        mEigAcl  = max(real(eig(Acl)));
        Y        = lyap(Acl',blkdiag(Q,Kt.Ck'*R*Kt.Ck));
        Jtemp    = trace(blkdiag(W,Kt.Bk*Kt.Bk')*Y);
    end

    if mod(Iter,Disp) == 0 || Iter == 1
        fprintf('%4d     %6.4E    %6.4E    %6.4E   %6.4E \n',Iter, ngradK, npar_gradK, J, StepSize);
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


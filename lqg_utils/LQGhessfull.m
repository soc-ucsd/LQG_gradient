function [Hess, info] = LQGhessfull(A,B,C,K1,Q,R,W,V)
% Compute the hessian of LQG at a given controller K1.
% use the natural cooridinate

[n,m] = size(B);
[p,n] = size(C);

Hess = zeros(n^2 + p*n + m*n);
Indx = ones(m+n,p+n);
Indx(1:m,1:p) = zeros(m,p);
Indx = Indx(:);

nonInd = find(Indx > 0);          % remove the top-left block
for i = 1:n^2 + p*n + m*n
    for j = i:n^2 + p*n + m*n   
        Delta1 = zeros((p+n)*(m+n),1);Delta1(nonInd(i)) = 1;
        Delta1 = reshape(Delta1,m+n,p+n); 
        Delta2 = zeros((p+n)*(m+n),1);Delta2(nonInd(j)) = 1;
        Delta2 = reshape(Delta2,m+n,p+n);  
        [Ja, Jb, Jc, H, info] = LQGhess(A,B,C,K1,Q,R,W,V,Delta1,Delta2);
        Hess(i,j) = H;
        Hess(j,i) = H;
    end
end
end


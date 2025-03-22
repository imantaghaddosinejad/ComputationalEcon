%% Solving linear system - Rendahl (2017) %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Linear Rational Expectations Model: Ax(t-1) + Bx(t) + Cx(t+1) + u(t) = 0
% where x is an n-vector of variables and u is mean zero iid disturbance.
% Assume a recursive solution of the form x(t) = Fx(t-1) + Qu(t) exists.
% This function iteratively solves for F and Q according to Rendahl (2017)
%
% Args:
%   A: (matrix) coefficients of x(t-1)
%   B: (matrix) coefficients of x(t) 
%   C: (matrix) coefficients of x(t+1) 
%   m: (scalar) small pertubation parameter 
%
% Returns:
%   F: (matrix) transition matrix for the system 
%   Q: (matrix) coefficient of linear disturbance term 
%
% System of equations to solve: C*F^2 + B*F + A = 0,-(B + C*F)^(-1) 
%
% Solve associated matrix polynomial: Ah*D^2 + Bh*D + Ch = 0 where
% Ah = C*(m^2*I) + B*(m*I) + A 
% Bh = B + C*(2m*I)
% Ch = C
% and m>0 is a (small) real number and I is a conformable identity matrix.
% 
% Iterattively solve for D: 
% D(n+1) = (Bh + Ah*D(n))^(-1) * (-Ch), which converges to Sh2 for initial guess D(0) != Sh1 
% D(n+1) = (Bh + Ch*D(n))^(-1) * (-Ah) converges to Sh1^(-1)
%
% Unique solution to original matrix polynomial: F = Sh1^(-1) + m*I
% % F is unique and stable solution if all eigenvalues (abs) of Sh1^(-1) 
% are inside the unit circle and all eigenvalues (abs) of Sh2 are inside 
% the unit circle 0 < m < 1 - max_i |eigvalue(Sh1^(-1))|
%
function [F, Q] = fnSolveLRE(A, B, C, m)
    mu = eye(size(A)) * m;
    Ah = C*mu.^2 + B*mu + A;
    Bh = B + C*(2.*mu);
    Ch = C;

    S2 = 0.01 .* ones(size(A));
    invS1 = 0.01 .* ones(size(A));
    S2new = zeros(size(invS1));
    invS1new = zeros(size(invS1));
    
    err = 10;
    MaxIter = 20000;
    iter = 1;
    while err > 1e-12 && iter <= MaxIter

        S2new = (Bh + Ah * S2) \ (-Ch);
        invS1new = (Bh + Ch * invS1) \ (-Ah);

        err1 = max(max(abs(S2new - S2)));
        err2 = max(max(abs(invS1new - invS1)));
        err = max(err1, err2);

        S2 = S2new;
        invS1 = invS1new;

        if mod(iter, 10) == 0
            fprintf('Error: %.12f\n', err);
        end
        iter = iter + 1;
    end
    F = invS1 + mu;
    Q = -inv(B + C * F); 
    
    [~, D1] = eig(invS1);
    [~, D2] = eig(S2);
    maxm = 1 - max(max(abs(D1)));

    if sum(max(abs(D1)) < 1) == size(invS1, 2) && sum(max(abs(D2)) < 1) == size(S2, 2)
        if m < maxm
            fprintf("Solution found (F) is unique and stable!")
        else 
            fprintf("m too large, bound it between (0,%.4f)", maxm)
        end
    elseif sum(max(abs(D1)) > 1) == size(invS1, 2) && sum(max(abs(D2)) < 1) == size(S2, 2)
        fprintf("Multiple stable solutions found!")
    else
        fprintf("No unique and stable solution found!")    
    end
end
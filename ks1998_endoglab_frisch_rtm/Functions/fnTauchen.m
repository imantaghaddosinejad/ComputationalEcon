%% Tauchen Discretization of AR(1) Process %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%   Discretize AR(1) Process
%
%   y(t) = mu + rho * y(t-1) + sigma * e(t), where e(t) ~ iid N(0, 1)
%   
%   stationary distribution: y(t) ~ N(mu/(1-rho), sigma^2 / (1-rho^2))
%   conditional distribution: y(t) | y(j) ~ N(mu+rho*y(j), sigma^2), for j<t
%   
%   Args:
%       rho: autocorrelation coefficient (must be bounded in (-1,1))
%       sigma: standard deviation of the stationary distribtuion of y(t)
%       mu: drift coefficient
%       n: number if discrete grid point
%       n_std: number of standard deviations for grid bounds (default: 3) 
%
%   Returns:
%       vector: [vZ, mP, dist] where 
%           vZ is a state vector, mP as transition matrix, dist is the invariant
%           distribution.
%       
function [vZ, P, dist] = fnTauchen(rho, sigma, mu, n, n_std, dist_true, eigval_true)
    if nargin < 5
        n_std = 3;
        dist_true = true;
        eigval_true = true;
    elseif nargin < 6
        dist_true = true;
        eigval_true = true;
    elseif nargin < 7
        eigval_true = true;
    end
    y_std = sigma / sqrt(1 - rho^2);
    y_mu = mu / (1 - rho);
    lb = y_mu - n_std * y_std;
    ub = y_mu + n_std * y_std;
    vZ = linspace(lb, ub, n)';
    w = ((ub-lb) / (n-1)) / 2 ;
    P = zeros(n, n);
    for j = 1:n
        for i = 1:n
            if j == 1
                P(i, j) = normcdf( ( vZ(j) + w - mu - rho*vZ(i) )  / sigma);
            elseif j == n
                P(i, j) = 1 - normcdf( ( vZ(j) - w - mu - rho*vZ(i) )  / sigma);                        
            else
                P(i, j) = normcdf( (vZ(j) + w - mu - rho * vZ(i)) / sigma) ...
                    - normcdf( (vZ(j) - w - mu - rho * vZ(i)) / sigma);                     
            end        
        end
    end
    if dist_true
        dist = fnStationaryDist(P, eigval_true);
    end
end

% Nested function to compute the stationary distribution
function vPi = fnStationaryDist(P, eigval_true)
    if eigval_true
        [v, D] = eig(P');
        [~, idx] = max(abs(diag(D)));
        vPi = v(:, idx);
        vPi = vPi / sum(vPi);
    else
        x0 = zeros(length(P), 1);
        x0(1) = 1;
        err = 10;
        while err >= 1e-12
            x1 = P' * x0;            
            err = max(abs(x1 - x0)); 
            x0 = x1;
        end
        vPi = x0;
    end
end
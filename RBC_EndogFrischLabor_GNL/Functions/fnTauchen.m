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
%       vector: [vZ, mP] where vZ is a state vector and mP as transition matrix
%       
function [vZ, P] = fnTauchen(rho, sigma, mu, n, n_std)
    if nargin < 5
        n_std = 3;
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
end
%% Optimization (Minimizaiton) Solver %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%   Find the Minimizer of a Function    
%
%   Args:
%       beta: (scalar) discount factor
%       budget:(scalar) total resources
%       grid: (vector) capital grid to interpolate kprime on
%       Ngrid: (scalar) total number of points in k-grid
%       expvf: (vector) expected value of VF (continuation value)
%       lb: (scalar) lower bound of minimizer
%       ub: (scalar) upper bound of minimizer
%
%   Returns:
%       kprime: (scalar) minimizer of objective function
%
function kprime = fnOptkprime(beta, budget, grid, Ngrid, expvf, lb, ub)
    options = optimset('Display', 'off', 'MaxIter', 2000, 'tolX', 1e-12, 'TolFun', 1e-12, 'MaxFunEvals', 20000);
    [kprime, ~] = fminbnd((@(x) fnHHobj(x, beta, budget, grid, Ngrid, expvf)), lb, ub, options);
end
%% Objective Funciton - Pass into a Solver/Minimizer %%%%%%%%%%%%%%%%%%%%%%
%
%   Objective Function with Interpolation 
%
%   Args:
%       x: (scalar) value to evaluate function at 
%       beta: (scalar) discount factor 
%       A: (scalar) TFP scale factor
%       budget: (scalar) total resources avaiable (upper constraint)
%       grid: (vector) grid to interpolate x on 
%       Ngrid: (scalar) number of gridpoints in grid 
%       expvf: (vector) expected value of VF (continuation value)
%
%   Returns:
%       val: (scalar) value of objective function at x
%
function val = fnObj(x, beta, budget, grid, Ngrid, expvf)
    kprime = x;
    c = max(1e-8, budget - kprime);
    [LB, UB, wtLB, wtUB] = fnInterp1dGrid(kprime, grid, Ngrid);
    Evalue = wtLB*expvf(LB) + wtUB*expvf(UB);
    val = -(log(c) + beta * Evalue);
end
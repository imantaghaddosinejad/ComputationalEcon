%% Household Objective Funciton - Pass into a Solver/Minimizer %%%%%%%%%%%%
function val = fnHHobj(x, beta, budget, grid, Ngrid, expvf)
%   Objective Function with Interpolation 
%
%   Args:
%       x: value to evaluate function at 
%       beta: discount factor 
%       budget: total resources avaiable (upper constraint)
%       grid: grid to interpolate x on 
%       Ngrid: number of gridpoints in grid 
%       expvf: expected value of value function (agent's continuation
%       value)
%
%   Returns:
%       val: scalar value
%
    aprime = x;
    c = max(1e-8, budget - aprime);
    [LB, UB, wtLB, wtUB] = fnInterp1dGrid(aprime, grid, Ngrid);
    Evalue = wtLB*expvf(LB) + wtUB*expvf(UB);
    val = -(log(c) + beta * Evalue);
end
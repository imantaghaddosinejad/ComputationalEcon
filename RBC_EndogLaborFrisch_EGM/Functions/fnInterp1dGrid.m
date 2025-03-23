%% Interpolation Funciton - 1D Grid %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%   Interpolate x on grid 
%   
%   Args:
%       x: numeric value to interpolate 
%       grid: grid to interpolate on
%       Ngrid: number of grid points on grid 
%
%   Returns:
%       vector: [low, high, wtlow, wthigh]
%
function [low, high, wtlow, wthigh] = fnInterp1dGrid(x, grid, Ngrid)
    low = sum(grid <= x);
    low = max(1, min(Ngrid - 1, low));
    high = low + 1;
    wtlow = (grid(high) - x) / (grid(high) - grid(low));   
    wtlow = max(0, min(1, wtlow)); % if x < grid, wtlow = 1, if x > grid, wtlow = 0
    wthigh = 1 - wtlow;
end
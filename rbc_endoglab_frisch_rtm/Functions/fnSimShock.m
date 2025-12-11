function path = fnSimShock(mTransMat, T, InitPoint, seed)
    mCDF = cumsum(mTransMat')'; % row-wise cum. prob. distribution
    path = zeros(T, 1);
    path(1) = InitPoint;
    
    if nargin < 4
        for t = 2:T
            path(t) = find(rand <= mCDF(path(t-1), :), 1, 'first');
        end
    else
        rng(seed);
        for t = 2:T
            path(t) = find(rand <= mCDF(path(t-1), :), 1, 'first');
        end    
    end
end
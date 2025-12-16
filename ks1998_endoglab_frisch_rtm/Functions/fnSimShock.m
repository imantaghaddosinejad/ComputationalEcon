function path = fnSimShock(mTransMat, T, InitPoint)
    mCDF = cumsum(mTransMat')'; % row-wise cum. prob. distribution
    path = zeros(T, 1);
    path(1) = InitPoint;
    for t = 2:T
        path(t) = find(rand <= mCDF(path(t-1), :), 1, 'first');
    end
end
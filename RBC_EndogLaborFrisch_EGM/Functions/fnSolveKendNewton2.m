function kend = fnSolveKendNewton2(k, z, c, n, kguess, params)
    
    % unpack parameters 
    alpha = params.palpha;
    delta = params.pdelta;

    % initial guess 
    kend = kguess;

    % newton root-finding iteration 
    err = 10;
    Tol = 1e-8;
    while err > Tol     
        % find optimal next value 
        f   = c + k - z*kend^alpha*n^(1-alpha) - (1-delta)*kend;
        df  = -alpha*z*kend^(alpha-1)*n^(1-alpha) - (1-delta);
        kp  = kend - (f/df);
        
        % compute error and update 
        err = abs(kp-kend);
        kend = kp;
    end
end
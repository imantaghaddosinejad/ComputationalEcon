function [kprime_opt, n_opt] = fnOptJoint2D(k, z, grid, Ngrid, expvf, params, mincap)

    % unpack parameters 
    beta    = params.pbeta;
    alpha   = params.palpha;
    delta   = params.pdelta;
    risk    = params.prisk;
    frisch  = params.pfrisch;
    eta     = params.peta;
    ssn     = params.ssn;

    % define objective function (nested function) 
    function val = fnObj(x)
        kp = x(1);
        n = x(2);
        c = z*k^alpha*n^(1-alpha) + (1-delta)*k - kp;
        if c <= 0 
            val = 1e12;
            return;
        end
        [LB, UB, wtLB, wtUB] = fnInterp1dGrid(kp, grid, Ngrid);
        Evalue = wtLB*expvf(LB) + wtUB*expvf(UB);
        value = c^(1-risk)/(1-risk) - eta*n^(1+ 1/frisch)/(1+1/frisch) + beta*Evalue;
        val = -value;
    end
    
    % set bounds  
    lb = [mincap, 0.001]; % avoid absolute 0 for numerical stability on wealth
    ub = [z*k^alpha + (1-delta)*k, 0.99];

    % initial guess 
    x0 = [0.99*k, ssn];

    % optimization options 
    options = optimoptions('fmincon','Display','off','Algorithm','sqp',...
        'MaxFunctionEvaluations',2000,'OptimalityTolerance',1e-8,'StepTolerance',1e-8);
    [x_opt, ~]  = fmincon(@fnObj, x0, [], [], [], [], lb, ub, [], options);
    kprime_opt  = x_opt(1);
    n_opt       = x_opt(2);

    % sanity check 
    c_opt = z*k^alpha*n_opt^(1-alpha) + (1-delta)*k - kprime_opt;
    if c_opt<=0
        warning('Optimal soln has negative consumption. Adjusting...');
        kprime_opt = 0.99 * (z*k^alpha*n_opt^(1-alpha) + (1-delta)*k);
    end
end
% SOLVING Barillas AND Fernandez-Villaverde (1972) GENERALIZED EGM %%%%%%%%
%
% 2024.03.22
% AUTHOR @ IMAN TAGHADDOSINEJAD (https://github.com/imantaghaddosinejad)
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Source: Barillas, F., Fernández-Villaverde, J. (2007). "A generalization 
% of the endogenous grid method." Journal of Economic Dynamics and Control
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% Algorithm:
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Housekeeping 
clear variables;
clc;
addpath('./Functions')

%% Model Fundamentals 

% parameters 
pbeta           = 0.9896;  
palpha          = 0.40; 
pdelta          = 0.0196;
priskaversion   = 2.00;
peta            = 7.60;
pfrisch         = 1.00;
Nk              = 300; % exogenous gridpoints - dense 
Nz              = 7; % productivity gridpoints 

% exog capital grid 
ssn = ((1-palpha)/peta)*((1/pbeta-1+pdelta)/palpha-pdelta)^(-priskaversion)*((1/pbeta-1+pdelta)/palpha)^((priskaversion-palpha)/(1-palpha));
ssk = ((1/pbeta-1+pdelta)/palpha)^(1/(palpha-1))*ssn;
ssc = (ssk/ssn)^(palpha-1) - pdelta;
kmin = 0.1*ssk;
kmax = 1.9*ssk;
% x = linspace(0,0.5,Nk);
% y = x.^1/max(x.^1);
% vgridk = kmin + (kmax-kmin)*y;
vgridk = linspace(kmin,kmax,Nk);step=diff(vgridk,1);step=step(1); % linear grid but dense
vgridk = vgridk'; % transpose for convenction 

% pack parameters (for functions)
params.pbeta=pbeta;params.palpha=palpha;params.pdelta=pdelta;params.prisk=priskaversion;params.pfrisch=pfrisch;params.peta=peta;params.ssn=ssn;

% productivity grid 
prho = 0.90;
pmu = 0.0;
psigma = 0.007;
[vgridz, mtransz] = fnTauchen(prho,psigma,pmu,Nz);
vgridz = exp(vgridz);

% exog resources grid given SS labor supply
vgridY = (vgridz'.*vgridk.^palpha).*ssn^(1-palpha) + (1-pdelta)*vgridk; 

%% Solving Stochastic Neoclassical Growth Model w/ Endog Labor Supply

% initial VF guess 
ssU = (1/(1-pbeta))*(ssc^(1-priskaversion)/(1-priskaversion) - peta*(ssn^(1+1/pfrisch)/(1+1/pfrisch)));
mVF = (ssU + (1:Nk)'/step) .* ones(1,Nz);

% equilibrium objects 
mVF_new     = zeros(size(mVF));
mPoln       = zeros(size(mVF)); 
mPolkprime  = zeros(size(mVF));
mPolc       = zeros(size(mVF));

%====================
% INITIAL EGM LOOP
%====================

iter = 1;
err = 10;
tol_egm = 1e-8;
tic;
while err > tol_egm

    % compute numerical derivative of VF over k'
    mdVF = zeros(size(mVF));
    mdVF(1,:) = (mVF(2,:) - mVF(1,:)) / (vgridk(2) - vgridk(1)); % Forward difference for first point
    mdVF(Nk,:) = (mVF(Nk,:) - mVF(Nk-1,:)) / (vgridk(Nk) - vgridk(Nk-1)); % Backward difference for last point
    for i = 2:Nk-1
        mdVF(i,:) = (mVF(i+1,:) - mVF(i-1,:)) / (vgridk(i+1) - vgridk(i-1)); % Central difference for interior points
    end

    % use the FOC to obtain contemporaneos consumption 
    ExpdVF = pbeta*(mdVF*mtransz'); 
    ctemp = (1./ExpdVF).^(1/priskaversion);
    
    % endogenous grid for resources 
    Yend = ctemp + vgridk;

    % endogenous VF over endogenous resource grid 
    VFend = ctemp.^(1-priskaversion)./(1-priskaversion) - peta*ssn.^(1+1/pfrisch)./(1+1/pfrisch) + pbeta*(mVF*mtransz');

    % interpolate VFend over fixed exogenous resource grid (defined over SS labor supply
    for iz = 1:Nz
        mVF_new(:,iz) = interp1(Yend(:,iz),VFend(:,iz),vgridY(:,iz),"linear","extrap");
    end

    % compute eror 
    err = max(abs(mVF_new - mVF),[],"all");

    % update 
    mVF = mVF_new;
    iter = iter + 1;
end
timer = toc;
fprintf('-------------------------------------------------- \n')
fprintf('Iter: %d    Time: %.3fs    Error: %.15f \n', iter, timer, err)

%====================
% ONE STEP VFI
%====================

mVF_new = zeros(size(mVF)); % reset placeholder for updated VF over exogenous capital-savings grid
iter = 1;
while iter <= 1 
    for iz = 1:Nz
        z = vgridz(iz); 
        ExpVF = mVF * mtransz';
        ExpVF = ExpVF(:,iz);
        MinWealth = (1-pdelta)*kmin;
        for ik = 1:Nk         
            % solve for optimal capital savings and labor supply
            k = vgridk(ik);
            mincap = MinWealth; % minimum value for optimal savings  
            [kprime, n] = fnOptJoint2D(k, z, vgridk, Nk, ExpVF, params, mincap);
            c = z*k^palpha*n^(1-palpha) + (1-pdelta)*k - kprime; 
            [LB, UB, wtLB, wtUB] = fnInterp1dGrid(kprime, vgridk, Nk);
            value = wtLB*ExpVF(LB) + wtUB*ExpVF(UB); % interpolate E[V]

            % updating
            mVF_new(ik,iz) = c^(1-priskaversion)/(1-priskaversion) - peta*n^(1+1/pfrisch)/(1+1/pfrisch) + pbeta*value;
            mPolc(ik,iz) = c;
            mPoln(ik,iz) = n;
            mPolkprime(ik,iz) = kprime;
            % MinWealth = kprime;
        end
    end
    iter = iter + 1;
end
err_vfi = max(abs(mVF_new - mVF),[],"all");

% impose frictions
mPolc(mPolc <= 1e-10) = 1e-10;
mPoln(mPoln < 0) = 0;
mPoln(mPoln > 1) = 1;

%====================
% PERIODIC VFI w/ INNER EGM LOOP
%====================

mVF = mVF_new; % set initial VF to the VF obtained in previous iteration of one-step VFI
tol_vfi = 1e-6;
iter_vfi = 1;
tic;
while err_vfi > tol_vfi 

    %====================
    % INTERPOLATE ENDOGENOUS GRID
    %====================
    
    % use a root-finding algorithm to back out endogenous grid 
    options = optimset('TolX',1e-6,'TolFun',1e-6);
    kend = zeros(Nk, Nz);
    for iz = 1:Nz
        for ik = 1:Nk
            fnKend = @(k) interp1(vgridk, mPolkprime(:,iz), k, "linear", "extrap") - vgridk(ik);
            kend(ik, iz) = fzero(fnKend, vgridk(ik), options);
        end
    end

    %====================
    % INTERPOLATE LABOR POLICY 
    %====================
    
    ntemp = zeros(size(mPoln));
    for iz = 1:Nz
        ntemp(:,iz) = interp1(vgridk, mPoln(:,iz), kend(:,iz), "linear", "extrap");
    end
    
    %====================
    % INNER EGM LOOP
    %====================

    err = 10;
    while err > 0.1*tol_vfi
        % compute numerical derivative of VF over k'
        mdVF = zeros(size(mVF));
        mdVF(1,:) = (mVF(2,:) - mVF(1,:)) / (vgridk(2) - vgridk(1)); % Forward difference for first point
        mdVF(Nk,:) = (mVF(Nk,:) - mVF(Nk-1,:)) / (vgridk(Nk) - vgridk(Nk-1)); % Backward difference for last point
        for i = 2:Nk-1
            mdVF(i,:) = (mVF(i+1,:) - mVF(i-1,:)) / (vgridk(i+1) - vgridk(i-1)); % Central difference for interior points
        end
        
        % compute contemporaneous consumption 
        ExpdVF = mdVF*mtransz';
        ExpdVF = pbeta*ExpdVF;
        ctemp = (1./ExpdVF).^(1/priskaversion);
        ctemp(ctemp<=1e-10)=1e-10;
    
        % given contemporaneous consumption backout endogenous grid holding labor at its previous EGM iteration level 
        kend = zeros(Nk,Nz);
        for iz = 1:Nz
            for ik = 1:Nk 
                kend(ik,iz) = fnSolveKendNewton2(vgridk(ik),vgridz(iz),ctemp(ik,iz),ntemp(ik,iz),vgridk(ik),params);
            end
        end
        
        % compute VF over endogenous capital grid  
        VFend = ctemp.^(1-priskaversion)./(1-priskaversion) - peta.*ntemp.^(1+1/pfrisch)./(1+1/pfrisch) + pbeta.*(mVF*mtransz');
        
        % interpolate updated VF over exogenous grid 
        for iz = 1:Nz
            mVF_new(:,iz) = interp1(kend(:,iz),VFend(:,iz),vgridk,"linear","extrap");
        end
        
        % update labor policy rule 
        % interpolate policy over new endogenous grid using the mapping from previous VFI iteration
        for iz = 1:Nz
            ntemp(:,iz) = interp1(vgridk,mPoln(:,iz),kend(:,iz),"linear","extrap");
        end
        
        % compute error and update VF  
        err = max(abs(mVF_new-mVF),[],"all");
        mVF = mVF_new;
    end

    %====================
    % ONE STEP VFI 
    %====================

    mVF_new = zeros(size(mVF)); % reset placeholder for updated VF over exogenous capital-savings grid
    iter = 1;
    while iter <= 1 
        for iz = 1:Nz
            z = vgridz(iz); 
            ExpVF = mVF * mtransz';
            ExpVF = ExpVF(:,iz);
            MinWealth = (1-pdelta)*kmin;
            for ik = 1:Nk         
                % solve for optimal capital savings and labor supply
                k = vgridk(ik);
                mincap = MinWealth; % minimum value for optimal savings  
                [kprime, n] = fnOptJoint2D(k, z, vgridk, Nk, ExpVF, params, mincap);
                c = z*k^palpha*n^(1-palpha) + (1-pdelta)*k - kprime; 
                [LB, UB, wtLB, wtUB] = fnInterp1dGrid(kprime, vgridk, Nk);
                value = wtLB*ExpVF(LB) + wtUB*ExpVF(UB); % interpolate E[V]
    
                % updating
                mVF_new(ik,iz) = c^(1-priskaversion)/(1-priskaversion) - peta*n^(1+1/pfrisch)/(1+1/pfrisch) + pbeta*value;
                mPolc(ik,iz) = c;
                mPoln(ik,iz) = n;
                mPolkprime(ik,iz) = kprime;
                % MinWealth = kprime;
            end
        end
        iter = iter + 1;
    end

    % compute error 
    err_vfi = max(abs(mVF_new - mVF),[],"all");
    
    % update VF 
    mVF = mVF_new;

    %====================
    % PROGRESS REPORTING
    %====================
    timer = toc;
    fprintf('-------------------------------------------------- \n')
    fprintf('Iter: %d    Time: %.3fs    Error: %.15f \n',iter_vfi,timer,err_vfi)   
    iter_vfi = iter_vfi + 1;
end



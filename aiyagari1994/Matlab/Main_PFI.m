% SOLVING AIYAGARI (1994) USING PFI %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% 2024.03.12
% Author @ Iman Taghaddosinejad (https://github.com/imantaghaddosinejad)
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% This file computes a stationary RCE using PFI with interpolation
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% This file incorporates techniques used by Hanbaek Lee
% Source: Lee, H. (2024) "Dynamically Consistent Global Nonlinear Solutions 
% in the Sequence Space:Theory and Applications."
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Outline of Algorithm: 
% 1. Parameterize
% 2. Guess aggregates and non-negativity constraint
% 3. Guess policy functions a'(a,z) and n(a,z) 
% 4. Interpolate on policy functions and compute beleifs 
% 5. Compute optimal allocations
% 6. Compute stationary distribution 
% 7. Update aggregates and policy rules 
% 8. Check convergence, if not repeat steps 4-8
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Housekeeping 
close all;clc;
clear variables;
% addpath('./Figures')
addpath('./Functions')

%% MODEL FUNDAMENTALS  
% model parameters 
p.Alpha     = 0.36;
p.Beta      = 0.99;
p.Delta     = 0.025;
p.Theta     = 0.50;
p.Frisch    = 1.00;
p.Eta       = 7.60;

% other parameters
p.Rhoz      = 0.90;
p.Sigmaz    = 0.05;
p.Nz        = 7;
p.Mina      = 1e-20;
p.Maxa      = 300;
p.Na        = 100;
p.Curve     = 7;

% unpack params 
pbeta=p.Beta;palpha=p.Alpha;pdelta=p.Delta;pfrisch=p.Frisch;peta=p.Eta;

%% GRIDS AND RANDOM PROCESSES 
% asset grid 
x = linspace(0,0.5,p.Na);
y = x.^p.Curve / max(x.^p.Curve);
vGrida = p.Mina + (p.Maxa - p.Mina).*y;

% idiosyncratic productivity grid
[vGridz, mPz] = fnTauchen(p.Rhoz,p.Sigmaz,0.0,p.Nz);
vGridz = exp(vGridz);

%% SRCE SETUP 
% auxillary objects 
mgrida = repmat(vGrida', 1, p.Nz); 
mgridz = repmat(vGridz', p.Na, 1);

% equilibrium objects 
mPolc           = repmat(0.01.*vGrida', 1, p.Nz);
K_new           = 0;
L_new           = 0;
mPolaprime_new  = zeros(size(mPolc)); 
mPoln_new       = zeros(size(mPolc));
mLambda_new     = zeros(size(mPolc));
A               = 1.0;

%  initial guess
K           = 11;
L           = 0.33;
mPolaprime  = ones(size(mPolc)); 
mPoln       = ones(size(mPolc)).*L;
mLambda     = zeros(size(mPolc));
mCurrDist   = ones(p.Na,p.Nz)/(p.Na*p.Nz);

% loop parameters 
tol_ge = 1e-8;
tol_pfi = 1e-8;
tol_dist = 1e-8;

% dynamic smoothing 
% given the non-linearity in the optimality equations and guessing two
% initial policy functions, convergence occurs in an osccilatory manner
% to avoid osscilation (computational accuracy isssue) smooting parameter
% is updated dynamically based on a weight schedule 
WtOld1_schedule = [0.990000, 0.999000, 0.999900, 0.999990, 0.999990, 0.999999, 0.999999];
WtOld2_schedule = [0.990000, 0.999000, 0.999900, 0.999990, 0.999990, 0.999999, 0.999999];
WtOld3_schedule = [0.900000, 0.900000, 0.990000, 0.990000, 0.999000, 0.999000, 0.999900];
WtOld4_schedule = [0.900000, 0.900000, 0.990000, 0.990000, 0.999000, 0.999000, 0.999900];
WtOld5_schedule = [0.900000, 0.900000, 0.990000, 0.990000, 0.999000, 0.999000, 0.999900];

%=========================
% GE LOOP
%=========================

% assign weights 
WtOld1 = WtOld1_schedule(1);
WtOld2 = WtOld2_schedule(1);
WtOld3 = WtOld3_schedule(1);
WtOld4 = WtOld4_schedule(1);
WtOld5 = WtOld5_schedule(1);
iwt = 1;

% initiate loop
iter1 = 1;
err1 = 10;
idisp_outer = 100;
%idisp_inner = 2500;
merror = {};
error_window = 500;
results = struct();
tic;
while err1 > tol_ge 

    %=========================
    % UPDATE PRICES
    %=========================

    r = palpha * A * (K/L)^(palpha-1) - pdelta;
    w = (1-palpha) * A * (K/L)^palpha;
    mmu = r+pdelta;

    %=========================
    % INNER LOOP - PFI 
    %%=========================

    iter2 = 1;
    err2 = 10;
    timer_temp = toc;
    while err2 > tol_pfi 

        %=========================
        % UPDATE BELIEFS 
        %=========================
    
        mExp = 0;
        for izprime = 1:p.Nz 
            
            % future realised state 
            rprime = r;
            wprime = w;
            zprime = vGridz(izprime);
            
            % interpolate policy rules  
            aprimeprime = interp1(vGrida', squeeze(mPolaprime(:,izprime)), mPolaprime, 'linear', 'extrap');
            nprime = interp1(vGrida', squeeze(mPoln(:,izprime)), mPolaprime, 'linear', 'extrap');
            
            % cumutitively compute expectation term 
            cprime = wprime.*zprime.*nprime + (1+rprime).*mPolaprime - aprimeprime;
            cprime(cprime < 1e-10) = 1e-10;
            muprime = 1./cprime;
            mExp = mExp + repmat(mPz(:,izprime)', p.Na, 1).*(1+rprime).*muprime;
        end
    
        %==========================
        % OPTIMAL POLICY GIVEN BELIEFS
        %==========================
        
        % intratemporal choice variables 
        mExp = pbeta*mExp;
        c = 1./(mExp+mLambda);
        c(c < 1e-10) = 1e-10;
        mPoln_new = ((w*mgridz)./(peta*c)).^pfrisch;
        mPoln_new(mPoln_new > 1) = 1;
        mPoln_new(mPoln_new < 0) = 0;
        
        % savings rule and lagrangian constraint 
        mLambda_new = 1./((1+r)*mgrida + w*mgridz.*mPoln - mPolaprime) - mExp;
        mPolaprime_new = (1+r)*mgrida + w*mgridz.*mPoln - c;
       
        % frictions 
        mLambda_new(mPolaprime_new>p.Mina) = 0;
        mPolaprime_new(mPolaprime_new<=p.Mina) = p.Mina;

        % compute error 
        err2 = abs(sum(mPolaprime_new.*mCurrDist, 'all') - sum(mPolaprime.*mCurrDist, 'all'));
        %err2 = max(abs(mPolaprime_new-mPolaprime), [], "all");
        err_n = max(abs(mPoln_new-mPoln), [], "all");
        err_lambda = max(abs(mLambda_new-mLambda), [], "all");

        % smooth updating 
        mPolaprime  = WtOld3.*mPolaprime    + (1-WtOld3).*mPolaprime_new;
        mPoln       = WtOld4.*mPoln         + (1-WtOld4).*mPoln_new;
        mLambda     = WtOld5.*mLambda       + (1-WtOld5).*mLambda_new;
        mPolc = c;

        % track progress
        % if mod(iter2, idisp_inner) == 0
        %     fprintf(' \n')
        %     fprintf('PFI LOOP \n')
        %     fprintf('iteration %d   error: %.15f \n', iter2, err2)
        %     fprintf('err_n: %.15f   err_lambda: %.15f \n', err_n, err_lambda)
        %     fprintf(' \n')
        % end
        iter2 = iter2+1;
    end
    timer1 = toc-timer_temp;

    %==========================
    % COMPUTE STATIONARY DISTRIBUTION
    %==========================

    % non-stochastic iterative histogram method
    iter3 = 1;
    err3 = 10;
    timer_temp = toc;
    while err3 > tol_dist

        mNewDist = zeros(size(mCurrDist));
        for iz = 1:p.Nz 
            for ia = 1:p.Na
                
                % for state (ia,iz) interpolate optimal savings on asset grid
                aprime = mPolaprime(ia,iz);
                nLow = sum(vGrida < aprime);
                nLow(nLow<=1) = 1;
                nLow(nLow>=p.Na) = p.Na - 1;
                nHigh = nLow + 1;
                wtLow = (vGrida(nHigh)-aprime)/(vGrida(nHigh)-vGrida(nLow));
                wtLow(wtLow>1) = 1;
                wtLow(wtLow<0) = 0;
                wtHigh = 1-wtLow;
                
                % get current mass over state (ia,iz) 
                mass = mCurrDist(ia,iz);
                
                % update mass cumutatively for all possible future iz states
                for izprime = 1:p.Nz
                    mNewDist(nLow, izprime) = mNewDist(nLow, izprime) + ...
                        1 * mass * mPz(iz,izprime) * wtLow;
                    mNewDist(nHigh, izprime) = mNewDist(nHigh, izprime) + ...
                        1 * mass * mPz(iz,izprime) * (wtHigh);
                end
            end
        end
        
        % compute error and update distribution 
        err3 = max(abs(mNewDist-mCurrDist),[],'all');
        mCurrDist = mNewDist;
        iter3 = iter3+1;
    end
    timer2 = toc-timer_temp;
    
    %==========================
    % COMPUTE AGGREGATES
    %==========================

    % market clearing 
    margdist            = sum(mCurrDist,2);
    K_new               = vGrida * margdist;
    L_new               = sum(mgridz.*mPoln.*mCurrDist, 'all');
    avgmLambda          = sum(mLambda.*mCurrDist, 'all'); 
    avgmPolaprime       = sum(mPolaprime.*mCurrDist, 'all');
    avgmPoln            = sum(mPoln.*mCurrDist, 'all');

    %==========================
    % UPDATING
    %==========================
    
    % error 
    err1 = mean(abs([...
        K_new - K;...
        L_new - L]),...
        'all'); 

    % smooth updating 
    K = WtOld1*K + (1-WtOld1)*K_new;
    L = WtOld2*L + (1-WtOld2)*L_new;
   
    %==========================
    % PROGRESS REPORTING
    %==========================

    timer3 = toc;
    if mod(iter1, idisp_outer) == 0 || iter1 == 1
        fprintf('-----------------------------------------------------------------\n');
        fprintf('market clearing results: \n') 
        fprintf('iteration %d in %.2fs    error: %.15f \n', iter1, timer3, err1);
        fprintf('K: %.12f \n', K)
        fprintf('L: %.12f \n', L)
        fprintf('r: %.12f \n', r)
        fprintf('w: %.12f \n', w)
        fprintf('min n: %.4f    max n: %.4f   min c: %.4f    max c: %.4f\n', min(min(mPoln)), max(max(mPoln)), min(min(mPolc)), max(max(mPolc)))
        fprintf('min_lambda: %.6f    max_lambda: %.6f \n', min(min(mLambda)), max(max(mLambda)))
        fprintf('Dist Converged in %d iterations in %.2fs \n', iter3, timer2)
        fprintf('PFI converged in %d iterations in %.2fs    iwt=%d \n', iter2, timer1, iwt)
    end
    
    % dynamically updating weighting parameter for smoother convergence 
    merror{end+1} = err1;
    if mod(iter1, 1000) == 0 && iter1 >= 1000 && iwt < 3
        iwt=iwt+1;
        %fprintf('Updating Weights! pct_pos=%.4f   New iwt=%d \n', pct_pos, iwt)
        WtOld1 = WtOld1_schedule(iwt);
        WtOld2 = WtOld2_schedule(iwt);
        WtOld3 = WtOld3_schedule(iwt);
        WtOld4 = WtOld4_schedule(iwt);
        WtOld5 = WtOld5_schedule(iwt);
    end

    if mod(iter1, 1000) == 0 && iter1 < 5000
        merror_array = cell2mat(merror);
        plot(merror_array(end-999:end)); grid on; ylabel('GE Error'); xlabel('Last 1,000 iterations'); drawnow; 
    elseif mod(iter1, 1000) == 0 && iter1 >= 5000 && iter1 < 10000
        merror_array = cell2mat(merror);
        plot(merror_array(end-4999:end)); grid on; ylabel('GE Error'); xlabel('Last 5,000 iterations'); drawnow; 
    elseif mod(iter1, 1000) == 0 && iter1 >= 10000 
        merror_array = cell2mat(merror);
        plot(merror_array(end-9999:end)); grid on; ylabel('GE Error'); xlabel('Last 10,000 iterations'); drawnow; 
    end
    iter1 = iter1 + 1;

    % Save Progress 
    results.wt_schedule1 = WtOld1_schedule;
    results.wt_schedule2 = WtOld3_schedule;
    results.K = K; results.L=L; results.w=w; results.r=r;
    results.err1=err1; results.iter1=iter1; results.timer=timer3;
    results.merror=merror; results.iwt=iwt;
    results.mPolapirime=mPolaprime; results.mPoln=mPoln; results.mPolc=mPolc; results.mLambda=mLambda;    
    save(".\results.mat", 'results')
end

%% Compute VF

mU = log(mPolc) - (peta/(1+1/pfrisch)).*mPoln.^(1+1/pfrisch);
mV = zeros(size(mU));
mVnew = 0;
err_vfi = 10;
iter_vfi = 1;
while err_vfi > 1e-10
    mVnew = mU + pbeta.*(mV*mPz');
    err_vfi = max(abs(mVnew-mV),[],"all");
    mV=mVnew;
end

%% Savings Rate and Wealth 
mWealthChange = mPolaprime-vGrida';
mNetIncome = r.*vGrida' + w.*vGridz'.*mPoln;
mSavingsRate = mWealthChange./mNetIncome;
mAvgSavingsRate = sum(mSavingsRate.*mCurrDist,2);

%% Figures 

figure;
plot(vGrida, mCurrDist);grid on;

figure;
plot(vGrida, mPolc); grid on;

figure;
plot(vGrida, mV); grid on;

figure;
plot(vGrida, sum(mCurrDist,2)); grid on;

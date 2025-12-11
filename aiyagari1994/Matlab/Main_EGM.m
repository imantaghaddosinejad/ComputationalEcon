% SOLVING AIYAGARI (1994) USING ENDOGENOUS GRID METHOD %%%%%%%%%%%%%%%%%%%%
%
% 2024.03.15
% Author @ Iman Taghaddosinejad (https://github.com/imantaghaddosinejad)
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% This file computes a stationary RE-RCE using Endogenous Grid Method
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Source: Carroll, C. D. (2006). "The method of endogenous gridpoints for 
% solving dynamic stochastic optimization problems." Economics letters.
% All mistakes are my own.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Outline of Algorithm:
% 1. Parameterization
% 2. Guess price(s) or Aggregate(s)
% 3. Solve HH optimizaiton problem using EGM
% 4. Compute staitonary distribution using non-stochastic itarative method
% 5. update price(s)/Aggregate(s)
% 6. Check convergence, if not repeat steps 3-6.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Housekeeping 
close all;clc;
clear variables;
addpath('./Figures')
addpath('./Functions')

%% MODEL FUNDAMENTALS  
% model parameters 
p.Alpha     = 0.36;
p.Beta      = 0.96;
p.Delta     = 0.080;

% other parameters
p.Rhoz      = 0.90;
p.Sigmaz    = 0.2 * sqrt(1 - p.Rhoz^2);
p.Nz        = 7;
p.Mina      = 0;
p.Maxa      = 300;
p.Na        = 100;
p.Curve     = 7;

% unpack params 
pbeta=p.Beta;palpha=p.Alpha;pdelta=p.Delta;

%% GRIDS AND RANDOM PROCESSES 
% asset grid 
x = linspace(0,0.5,p.Na);
y = x.^p.Curve / max(x.^p.Curve);
vGrida = p.Mina + (p.Maxa - p.Mina).*y;

% idiosyncratic productivity grid
[vGridz, mPz] = fnTauchen(p.Rhoz,p.Sigmaz,0.0,p.Nz);
vGridz = exp(vGridz);
vDistz = fnStationaryDist(mPz);

%% SOLVE MODEL
% auxillary objects 
mgrida = repmat(vGrida', 1, p.Nz); 
mgridz = repmat(vGridz', p.Na, 1);

% equilibrium objects 
K_new       = 0;
A           = 1.0;
mPolc_new   = zeros(p.Na,p.Nz);

% initial guess 
K           = 7;
L           = vGridz' * vDistz;
r           = palpha*A*(K/L)^(palpha-1) - pdelta;
w           = (1-palpha)*A*(K/L)^palpha;
mPolc       = (1+r)*mgrida + w*mgridz - mgrida; % initial consumption policy 
mCurrDist   = ones(p.Na,p.Nz)/(p.Na*p.Nz);

% loop parameters 
tol_ge = 1e-8;
tol_egm = 1e-8;
tol_dist = 1e-8;
wtOld1 = 0.99000; % aggregate allocation (K) updating smoothness parameter
wtOld2 = 0.90000; % policy rule (c) updating smoothness parameter

% start GE loop
iter1 = 1;
err1 = 10;
merror = {};
tic;
while err1 > tol_ge

    %====================
    % UPDATE PRICES
    %====================

    r = palpha*A*(K/L)^(palpha-1) - pdelta;
    w = (1-palpha)*A*(K/L)^palpha;

    %====================
    % INNER LOOP - EGM 
    %====================

    mBudget = (1+r)*mgrida + w*mgridz;
    mgrida_endog = zeros(p.Na, p.Nz);
    iter2 = 1;
    err2 = 10;
    timer_temp = toc;
    while err2 > tol_egm
        
        % compute expectation (beliefs)
        rprime = r;
        Exp = pbeta * (1+rprime) * ((1./mPolc) * mPz');
        
        % contemporaneous consumption 
        ctemp = 1./Exp;

        % back out endogenous grid 
        mgrida_endog = (ctemp + mgrida - w*mgridz) / (1+r);
        
        % update savings policy 
        mPolaprime_new = zeros(p.Na,p.Nz);
        for iz = 1:p.Nz
            mPolaprime_new(:,iz) = interp1(mgrida_endog(:,iz), vGrida', vGrida', "linear", "extrap");
            for ia = 1:p.Na 
                if mPolaprime_new(ia,iz) < p.Mina
                    mPolaprime_new(ia,iz) = p.Mina; % apply friction - lower bound constraint
                else
                    break % exploit monotonicity of policy rule 
                end
            end
        end

        % compute consumption on exogenous grid 
        mPolc_new = mBudget - mPolaprime_new;

        % compute error 
        err2 = max(abs(mPolc_new - mPolc),[],"all");

        % update policy rules 
        mPolc = wtOld2*mPolc + (1-wtOld2)*mPolc_new;
        mPolc(mPolc < 1e-10) = 1e-10;
        mPolaprime = mPolaprime_new;
        iter2 = iter2+1;
    end
    timer2 = toc - timer_temp;

    %====================
    % STATIONARY DISTRIBUTION
    %====================

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
    timer3 = toc-timer_temp;

    %====================
    % COMPUTE AGGREGATES
    %====================

    % market clearing 
    margdist    = sum(mCurrDist,2);
    K_new       = vGrida * margdist;
  
    %====================
    % UPDATING
    %====================
    
    % error 
    err1 = abs(K_new - K); 
    merror{end+1}=err1;

    % smooth updating 
    K = wtOld1*K + (1-wtOld1)*K_new;
    
    %====================
    % REPORT PROGRESS
    %====================

    timer1 = toc;
    if mod(iter1, 50) == 0 || iter1 == 1
        fprintf('--------------------------------------------------------- \n');
        fprintf('market clearing results: \n') 
        fprintf('iteration %d in %.2fs    error: %.15f \n', iter1, timer1, err1);
        fprintf('K: %.12f \n', K)
        fprintf('r: %.12f \n', r)
        fprintf('w: %.12f \n', w)
        fprintf('min aprime: %.4f    max aprime: %.4f   min c: %.4f    max c: %.4f\n', ...
            min(min(mPolaprime)), max(max(mPolaprime)), min(min(mPolc)), max(max(mPolc)))
        fprintf('EGM converged. iter: %d    Time: %.2fs \n', iter2, timer2)
        fprintf('Dist converged. iter: %d   Time: %.2fs \n', iter3, timer3) 
        %fprintf('wtOld1: %.5f \n', wtOld1)
    end

    % dynamic update weights smoothing 
    % if iter1 >= 1000 && iter1 < 10000
    %     wtOld1 = 0.99900;
    % % else
    % %     wtOld1 = 0.99990;
    % end
    iter1 = iter1 + 1;
end

%% solution analysis

% savings rate 
mSR = (r*mgrida + w*mgridz - mPolc)./(r*mgrida + w*mgridz);

% error path 
err_array = cell2mat(merror);

% marginal propensity to consume over wealth (adjusted/normalized by wealth level)
my = r*mgrida+w*mgridz;
ms = mPolaprime - mgrida;
mc_adj = mPolc./mgrida;
ms_adj = ms./mgrida;
my_adj = my./mgrida;
dy = diff(my_adj,1,1);
dc = diff(mc_adj,1,1);
ds = diff(ms_adj,1,1);
mpc_adj = dc./dy;
mps_adj = ds./dy;

% plots 
subplot(2,3,1); 
plot(vGrida, sum(mCurrDist,2), 'LineWidth',1); grid on;xlabel('a');title('marginal dist a');xlim([0 120])

subplot(2,3,2);
plot(vGrida,mPolc);grid on;xlabel('a');title('c(a,z)');xlim([0 120])

subplot(2,3,3);
plot(vGrida, mPolaprime); grid on;xlabel('a');title("a'(a,z)");xlim([0 120])

subplot(2,3,4);
plot(vGrida, mCurrDist,'LineWidth',0.8); grid on;xlabel('a');title('dist a over (a,z)');xlim([0 120])

subplot(2,3,5);
plot(vGrida,mSR,'LineWidth',1); grid on;xlabel('a');title('savings rate');xlim([0 120])

subplot(2,3,6);
%plot(err_array(end-2000:end),'LineWidth',1); grid on;xlabel('last 2k iteraitons');title('error (K)')
plot(vGrida(2:end), mpc_adj,'LineWidth',1);grid on;xlabel('a');title('MPC (relative to wealth)'); xlim([0 120]);legend('z1','z2','z3','z4','z5','z6','z7');
saveas(gcf,'./Figures/Model_Results.png');

figure;
plot(vGrida(2:end), mps_adj,'LineWidth',1);grid on;xlabel('x');title('MPS (relative to wealth)'); xlim([0 120]);legend('z1','z2','z3','z4','z5','z6','z7');
saveas(gcf,'./Figures/MPS.png')

figure;
plot(err_array(end-850:end),'LineWidth',1); grid on;xlabel('last 850 iteraitons');title('error (K)')
saveas(gcf,'./Figures/error_gr_path.png')
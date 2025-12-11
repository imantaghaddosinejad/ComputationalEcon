%% SOLVING BROCK AND MIRMAN (1972) USING PFI %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% 2024.03.12
% AUTHOR @ IMAN TAGHADDOSINEJAD (https://github.com/imantaghaddosinejad)
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% This file computes optimal policy functions using PFI with interpolation
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Outline of Algorithm:
% 1. Parameterize
% 2. Guess policy k'(k,z)
% 3. Interpolate on k'(k,z) to compute expectations
% 4. Update policy 
% 5. Check convergence, if not repeat steps 3-5
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Housekeeping 
clear 
clc 
addpath('./Functions')

%% Set Parameters

% Model Parameters 
pAlpha = 0.36;
pBeta = 0.96;
pA = 1.0;
pKss = (1/(pBeta*pAlpha))^(1/(pAlpha-1));
pKmin = 0.1 * pKss;
pKmax = 1.9 * pKss;

% grid (k,z) parameters 
pMu = 0;
pRho = 0.90;
pSigma = 0.0870;
pNz = 4;
pNk = 200;
pCurve = 3;

% capital grid - (Maliar, Maliar and Valli, 2010)
x = linspace(0, 0.5, pNk);
y = x.^pCurve / max(x.^pCurve);
vGridk = pKmin + (pKmax - pKmin)*y;

% productivity grid 
[vZ, mPz] = fnTauchen(pRho, pSigma, pMu, pNz);
vGridz = exp(vZ);

% auxillary objects 
mz = repmat(vGridz',pNk,1);
mk = repmat(vGridk',1,pNz);

%% Solve Model

% initial guess (policy function)
mPolkprime = repmat(0.01*vGridk', 1, pNz);

% equilibrium objects 
mPolc = zeros(size(mPolkprime));
mPolkprime_new = zeros(size(mPolkprime));
%minCapital = pkmin; 

% inititalise loop
iter = 1; 
MaxIter = 2000;
errPF = 10;
errPF_array = {};
errTolPF = 1e-10;
wtOld = 0.9500;
tic;
while errPF > errTolPF && iter <= MaxIter 

    %====================
    % EXPECTAITON TERM 
    %====================

    mExp = 0;
    for izprime = 1:pNz 

        zprime = vGridz(izprime);

        % interpolate on last iteration policy rule over all (ik,iz) states
        kprimeprime = interp1(vGridk', squeeze(mPolkprime(:,izprime)),mPolkprime,"linear","extrap");

        % compute future consumption given (ik',iz')
        cprime = zprime.*mPolkprime.^(pAlpha) - kprimeprime;
        cprime(cprime < 1e-10) = 1e-10;

        % cumulatively update expectation term 
        muprime = 1./cprime;
        mExp = mExp + repmat(mPz(:,izprime)',pNk,1) .* pAlpha*zprime.*mPolkprime.^(pAlpha-1) .* muprime;
    end

    % optimal allocations given beliefs 
    mExp = pBeta.*mExp;
    c = 1./mExp;
    
    %====================
    % UPDATING POLICY
    %====================

    % new rules  
    mPolkprime_new = mz.*mk.^pAlpha - c;
    mPolkprime_new(mPolkprime_new <= 0) = 1e-10;
    mPolc = c;
    mPolc(mPolc <= 0) = 1e-10;

    % compute error 
    errPF = max(max(abs(mPolkprime_new - mPolkprime)));
    errPF_array{end + 1} = errPF;

    % updating 
    mPolkprime = wtOld.*mPolkprime + (1-wtOld).*mPolkprime_new;
    
    %====================
    % PROGRESS REPORTING
    %====================
    
    timer = toc;
    if mod(iter, 25) == 0
    fprintf('Iter: %d in %.2fs  Error: %.15f \n', iter, timer, errPF)
    fprintf('------------------------------ \n')
    err_array = cell2mat(errPF_array);
    plot(err_array(25:end),'LineWidth',1); grid on; drawnow;
    end
    iter = iter + 1;
end

%% Iteratively Compute Value Function
mU = log(mPolc);
mV = zeros(size(mU));
mV_new = 0;
iter = 1;
err = 10;
while err > 1e-8 
   mV_new =  mU + pBeta*mV*mPz';
   err = max(max(abs(mV_new - mV)));
   mV = mV_new;
end

% savings rate 
my = mPolc + mPolkprime;
msr = mPolkprime./my;

% MPC out of wealth 
dy = diff(my,1,1);
dc = diff(mPolc,1,1);
MPC = dc./dy;
MPS = 1-MPC;

%% Plots and Figures 

% capital policy function
figure;
plot(vGridk, mPolkprime, 'LineWidth', 1.5)
hold on;
plot(vGridk,vGridk,'k--','LineWidth',1)
hold off;
xlim([pKmin pKmax])
xlabel('k(t)', 'FontSize', 15)
ylabel('k(t+1)', 'FontSize', 15)
grid on;
legend('z(1)', 'z(2)', 'z(3)', 'z(4)', 'Location', 'northwest')
saveas(gcf, './Figures/KprimePol_pfi.png');

% consumption policy function
figure;
plot(vGridk, mPolc, 'LineWidth', 1.5)
hold on;
plot(vGridk,vGridk,'k--','LineWidth',1)
hold off;
xlabel('k(t)', 'FontSize', 15)
ylabel('c(t)', 'FontSize', 15)
grid on;
xlim([pKmin pKmax])
legend('z(1)', 'z(2)', 'z(3)', 'z(4)', 'Location', 'northwest')
saveas(gcf, './Figures/cPol_pfi.png');

% value function 
figure;
plot(vGridk, mV, 'LineWidth', 1.5); grid on;xlim([pKmin pKmax])
xlabel('k(t)', 'FontSize', 15)
ylabel('VF', 'FontSize', 15)
legend('z(1)', 'z(2)', 'z(3)', 'z(4)', 'Location', 'northwest')
saveas(gcf, './Figures/VF_pfi.png');
%% Solving Brock and Mirman Model (1972) %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% 2024.12.21
% Author @ Iman Taghaddosinejad (https://github.com/iman-nejad)
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% This file computes optimal policy functions using VFI with interpolation
% and Howard improvement
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Outline of Algorithm:
% 1. Parameterization (model setup)
% 2. Guess initial VF
% 3. Solve HH optimizaiton problem with interpolation + Howard improvement 
% 4. Update VF (and Policy rules)
% 5. Check convergence, if not repeat steps 3-5.
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
pKss = (pBeta * pAlpha * pA)^(1-pAlpha);
pKmin = 0.1 * pKss;
pKmax = 1.9 * pKss;

% grid (k,z) parameters 
pMu = 0;
pRho = 0.90;
pSigma = 0.0870;
pNz = 4;
pNk = 100;
pCurve = 2;

% capital grid - (Maliar, Maliar and Valli, 2010)
x = linspace(0, 0.5, pNk);
y = x.^pCurve / max(x.^pCurve);
vGridk = pKmin + (pKmax - pKmin)*y;

% productivity grid 
[vZ, mPz] = fnTauchen(pRho, pSigma, pMu, pNz);
vGridz = exp(vZ);

%% Solve Model

% initialise VFI loop
mVF = repmat(0.01 * vGridk', 1, pNz);
mVFnew = zeros(size(mVF));
mPolkprime = zeros(size(mVF));
mPolc = zeros(size(mVF));
minCapital = pKmin;

% loop parameters 
iter = 1; 
MaxIter = 1000;
errVF = 10;
errTolVF = 1e-8;

% VFI w/ Howard Improvement and Interpolation of Policy Function
%figure
tic;
while errVF > errTolVF && iter <= MaxIter    
    for iz = 1:pNz
        z = vGridz(iz);
        expVF = mVF * mPz';
        expVF = expVF(:, iz); % expected VF given current iz state
        for ik = 1:pNk
            %----- VF COMPUTATION w/ POLICY FUNCTION (NON-ACCELERATED) -----%
            if mod(iter, 20) == 0 || iter <= 30
                k = vGridk(ik);
                budget = z*pA*(k^pAlpha);
                lb = minCapital;
                ub = budget;
                kprime = fnOptkprime(pBeta, budget, vGridk, pNk, expVF, lb, ub);
                c = budget - kprime;
    
                % interpolate kprime on the grid and hence interpolate VF
                [LB, UB, wtLB, wtUB] = fnInterp1dGrid(kprime, vGridk, pNk);
                value = wtLB*expVF(LB) + wtUB*expVF(UB);
    
                % updating 
                mVFnew(ik, iz) = log(c) + pBeta*value;
                mPolkprime(ik, iz) = kprime;
                mPolc(ik, iz) = c;
            %----- VF COMPUTATION w/o POLICY FUNCTION (ACCELERATED) -----%
            else
                kprime = mPolkprime(ik, iz);
                c = mPolc(ik, iz);
                [LB, UB, wtLB, wtUB] = fnInterp1dGrid(kprime, vGridk, pNk);
                value = wtLB*expVF(LB) + wtUB*expVF(UB);
                mVFnew(ik, iz) = log(c) + pBeta*value;
            end
        end
    end    
    % compute error 
    errVF = max(max(abs(mVFnew - mVF)));
    
    % plotting VF Convergence  
    % if mod(iter, 20) == 0
    %     plot(vGridk, mVF(:, 2), 'LineWidth', 1.5, 'Color', 'b')
    %     hold on;
    %     plot(vGridk, mVFnew(:, 2), 'LineWidth', 1.5, 'LineStyle', '--', 'Color', 'r')
    %     hold off;
    %     xlim([pKmin pKmax])
    %     grid on;
    %     xlabel('k', 'FontSize', 18)
    %     ylabel('V(k,z)', 'FontSize', 18)
    %     legend('VF(n)', 'VF(n+1)', 'Location', 'southeast')
    %     drawnow;
    % end

    % update VF
    mVF = mVFnew;
    % progress reporting
    timer = toc;
    if mod(iter, 40) == 0
        fprintf('Iteration: %d in %.2fs. Error: %.8f\n', iter, timer, errVF)
        fprintf('--------------------------------------------------\n')
    end
    iter = iter + 1;
end

%% Figures 

% Value Function
figure;
plot(vGridk, mVF(:, 2), 'LineWidth', 1.5)
grid on;
xlim([pKmin pKmax])
xlabel('k', 'FontSize', 15)
ylabel('V(k,z)', 'FontSize', 15)
legend('z = z(2)', 'Location', 'southeast')
saveas(gcf, './Figures/ValueFunction.png');

% capital policy function
figure;
plot(vGridk, mPolkprime, 'LineWidth', 1.5)
hold on;
plot(vGridk, vGridk, 'LineStyle', '--', 'LineWidth', 1, 'Color', 'k')
hold off;
xlim([pKmin pKmax])
ylim([pKmin pKmax])
xlabel('k(t)', 'FontSize', 15)
ylabel('k(t+1)', 'FontSize', 15)
grid on;
legend('z(1)', 'z(2)', 'z(3)', 'z(4)', 'Location', 'northwest')
saveas(gcf, './Figures/KprimePol.png');

% consumption policy function
figure;
plot(vGridk, mPolc, 'LineWidth', 1.5)
hold on;
plot(vGridk, vGridk, 'LineStyle', '--', 'LineWidth', 1, 'Color', 'k')
hold off;
xlabel('k(t)', 'FontSize', 15)
ylabel('c(t)', 'FontSize', 15)
grid on;
xlim([pKmin pKmax])
ylim([pKmin 1.201])
legend('z(1)', 'z(2)', 'z(3)', 'z(4)', 'Location', 'northwest')
saveas(gcf, './Figures/cPol.png');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
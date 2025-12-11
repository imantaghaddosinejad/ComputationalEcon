%% Solving Aiyagari Model (1994) %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% 2024.12.11
% Author @ Iman Taghaddosinejad (https://github.com/iman-nejad)
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% This file computes a stationary RE-RCE using VFI with Interpolated 
% search + Howard Improvement
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Outline of Algorithm:
% 1. Parameterization
% 2. Guess price(s) or Aggregate(s)
% 3. Solve HH optimizaiton problem using VFI + Howard Improvement 
% 4. Compute staitonary distribution using non-stochastic itarative method
% 5. update price(s)/Aggregate(s)
% 6. Check convergence, if not repeat steps 3-6.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Housekeeping 
clear 
clear all 
clc
addpath('./Functions')

%% Set Parameters 

% Aiyagari (1994) 
pAlpha = 0.36;
pBeta = 0.96;
pDelta = 0.08;
pTFP = 1;

% income process 
pMu = 0;
pRho = 0.90;
pSigma = 0.2 * sqrt(1 - pRho^2);
pNz = 7;

% wealth grid 
pNa1 = 50;
pNa2 = 100;
pMinGrida = 0;
pMaxGrida = 150;
pCurve = 7;

% loop parameters 
pWtOldK = 0.9900;
pTolGE = 1e-8;
pTolVF = 1e-8;
pTolDist = 1e-8;
pMaxGEiter = 2000;
pMaxVFiter = 2000;
convergedGE = false;
convergedVF = false;

%% Define Grids 

% wealth - coarsed grid (Maliar, Maliar and Valli, 2010)
x1 = linspace(0, 0.5, pNa1);
x2 = linspace(0, 0.5, pNa2);
y1 = x1.^pCurve / max(x1.^pCurve);
y2 = x2.^pCurve / max(x2.^pCurve);
vGrida1 = pMinGrida + (pMaxGrida - pMinGrida) * y1;
vGrida2 = pMinGrida + (pMaxGrida - pMinGrida) * y2;

% productivity 
[vZ, mPz] = fnTauchen(pRho, pSigma, pMu, pNz);
vPiz = fnStationaryDist(mPz);
vGridz = exp(vZ);

%% Initialize 

% initial guess (aggregate(s)) 
aggKinit = 6;
mVF = repmat(0.01 .* vGrida1', 1, pNz);
mCurrDist = ones(pNa2, pNz);
mCurrDist = mCurrDist ./ (pNa2 * pNz);

% placeholders 
mVFnew = zeros(pNa1, pNz);
mPolaprime1 = zeros(pNa1, pNz);
mPolc = zeros(pNa1, pNz);
aggKnew = 0;
%mPolaprime2 = zeros(size(mCurrDist));
%mNewDist = zeros(pNa2, pNz);

%% Solve Model

aggK = aggKinit;
aggL = vGridz' * vPiz; 
iterGE = 1;
errGE = 20;
tic; 
while errGE > pTolGE && iterGE <= pMaxGEiter

    r = pAlpha * pTFP * (aggK / aggL)^(pAlpha - 1) - pDelta;
    w = (1 - pAlpha) * pTFP * (aggK / aggL)^pAlpha;

    % --------------------------------------------------- %
    % INNER LOOP - VFI + HOWARD IMPROVEMENT
    % --------------------------------------------------- %

    iterVF = 1;
    errVF = 20;
    while errVF > pTolVF && iterVF <= pMaxVFiter
                
        for iz = 1:pNz

            z = vGridz(iz);
            expVF = mVF * mPz'; % expected future value over all current states 
            expVF = expVF(:, iz); % expected future value conditional on current state iz
            minWealth = pMinGrida; % reset lower bound for current state 
    
            for ia = 1:pNa1
                
                % ----- VFI w/ Interpolated Policy Function ----- %
        
                if iterVF <= 30 || mod(iterVF, 20) == 0
                    a = vGrida1(ia);
                    budget = w*z + (1+r)*a; 
                    lb = minWealth;
                    ub = budget;
                    aprime = fnOptaprime({pBeta, budget, vGrida1, pNa1, expVF}, lb, ub);
                    c = budget - aprime;
    
                    [LB, UB, wtLB, wtUB] = fnInterp1dGrid(aprime, vGrida1, pNa1);
                    value = wtLB*expVF(LB) + wtUB*expVF(UB); % interpolate E[V]
                    
                    % updating
                    mVFnew(ia, iz) = log(c) + pBeta * value;
                    mPolaprime1(ia, iz) = aprime;
                    mPolc(ia, iz) = c;
                    minWealth = aprime; % exploit monotonicity of policy rule
                
                % ----- VFI w/ Accelerated Howard Improvement ----- % 
                
                else    
                    aprime = mPolaprime1(ia, iz);
                    c = mPolc(ia, iz);
                    [LB, UB, wtLB, wtUB] = fnInterp1dGrid(aprime, vGrida1, pNa1);
                    value = wtLB*expVF(LB) + wtUB*expVF(UB); 
                    
                    % updating
                    mVFnew(ia, iz) = log(c) + pBeta * value;
                end
            end
        end
    
        errVF = max(max(abs(mVFnew - mVF)));
        mVF = mVFnew;
        
        %if mod(iterVF, 50) == 0
        %    fprintf('Iteration: %d, Error: %.6f\n', iterVF, errVF);
        %end        
    
        iterVF = iterVF + 1;
    end
    
    % --------------------------------------------------- %
    %  COMPUTE STATIONARY DISTRIBUTION 
    % --------------------------------------------------- %
    
    % interpolate aprime policy function onto a finer grid 
    mPolaprime2 = interp1(vGrida1, mPolaprime1, vGrida2, 'linear', 'extrap');
    %mPolaprime2 = zeros(size(mCurrDist));
    %for iz = 1:pNz
    %    mPolaprime2(:, iz) = interp1(vGrida1, mPolaprime1(:, iz), vGrida2, "linear", "extrap");
    %end

    errDist = 20;
    iterDist = 1;
    while errDist > pTolDist

        mNewDist = zeros(size(mCurrDist)); % reset distribution to incremeentally increase mass over states (ia,iz)

        for iz = 1:pNz
            for ia = 1:pNa2    
                aprime = mPolaprime2(ia, iz);
                [L, H, wtL, wtH] = fnInterp1dGrid(aprime, vGrida2, pNa2);
                mass = mCurrDist(ia, iz);
                
                % updating
                for iznext = 1:pNz
                    mNewDist(L, iznext) = mNewDist(L, iznext) ...
                        + mPz(iz, iznext) * mass * wtL;

                    mNewDist(H, iznext) = mNewDist(H, iznext) ...
                        + mPz(iz, iznext) * mass * wtH;
                end
            end
        end

        errDist = max(max(abs(mNewDist - mCurrDist)));
        mCurrDist = mNewDist;
        iterDist = iterDist + 1;

        %if mod(iterDist, 20) == 0
        %    fprintf('Iteration: %d, Error: %.6f\n', iterDist, errDist);
        %end
    end

    % --------------------------------------------------- %
    %  UPDATE AGGREGATES USING MARKET CLEARING 
    % --------------------------------------------------- %
    
    vMarginalDista = sum(mCurrDist, 2); % (marginal) density over asset-grid
    Kmcc = vGrida2 * vMarginalDista; % capital market clearing condition
    errGE = abs(Kmcc - aggK);

    % updating
    aggKnew = pWtOldK*aggK + (1-pWtOldK)*Kmcc; % smooth updating     
    aggK = aggKnew; 
    timer = toc;

    % --------------------------------------------------- %
    %  REPORT PROGRESS 
    % --------------------------------------------------- %
    
    if mod(iterGE, 10) == 0
        fprintf('Iteration: %d in %.3f mins. Error: %.8f\n', iterGE, timer./60, errGE);
        fprintf('AggK: %.4f\n', aggK);
        fprintf('Ea(r): %.4f\n', Kmcc)
        fprintf('r: %.4f, w: %.4f\n', r, w);
        fprintf('----------------------------------------------------\n')
    end
    iterGE = iterGE + 1;
end

% Convergence Report 
if errGE <= pTolGE
    convergedGE = true;
end
fprintf('GE Converged: %s\n', mat2str(convergedGE));

%% Figures 

% plot marignal density of wealth 
plot(vGrida2, vMarginalDista, 'LineWidth', 2);
xlabel('Wealth (a)');  
ylabel('Density');  
grid on;  
saveas(gcf, './Figures/WealthDist.png');

% plot wealth distribution over productivity (iz) states     
hold on;
for iz = 1:pNz
    plot(vGrida2, mCurrDist(:, iz), 'LineWidth', 1.3);
end
xlabel('Wealth (a)');
ylabel('Density');
grid on;
legend(arrayfun(@(iz) sprintf('z = %.2f', vGridz(iz)), 1:size(mCurrDist, 2), 'UniformOutput', false));
hold off;
saveas(gcf, './Figures/WealthDistOverZ.png');

% plot value function over wealth and policy function over wealth 
legendlab1 = arrayfun(@(iz) sprintf('z = %.2f', vGridz(iz)), 1:size(mCurrDist, 2), 'UniformOutput', false);
legendlab2 = {sprintf('z = %.2f', vGridz(1)), sprintf('z = %.2f', vGridz(7))};

figure; 

subplot(1, 2, 1);
plot(vGrida1, mVF, 'LineWidth', .8);
ylabel('V(a,z)');
xlabel('assets');
legend(legendlab1, 'Location', 'southeast')
grid on;

subplot(1, 2, 2);
plot(vGrida2, mPolaprime2(:, 1), 'LineWidth', .8)
hold on;
plot(vGrida2, mPolaprime2(:, 7), 'Linewidth', .8);
hold off;
ylabel("Policy Rule (g(a,z) = a')");
xlabel("assets");
legend(legendlab2, 'Location', 'southeast')
grid on;
saveas(gcf, './Figures/VF_and_PFaprime.png');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

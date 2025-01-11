%% Solving Canonical RBC Model Using Nonlinear Global Solution Method %%%%%
%
% 2025.01.11
% Author @ iman Taghaddosinejad (github.com/imantaghaddosinejad)
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% This file computes a RCE for a RBC model with enod. labor choice using a 
% nonlinear global solution method in sequence space. This version of the 
% code uses a vectorization approach instead of time iteration to boost 
% computation speed. 
%
% Reference: Lee., H. (2022), "A Dynamically Consistent Global 
% Nonlinear Solution Method in the Sequence Space and Applications."
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% This file incorporates techniques used by Hanbaek Lee in his code 
% (github.com/hanbaeklee/ComputationLab/tree/main/rbcfrischlabor). 
% All mistakes are my own.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Outline of Algorithm:
% 1. Parameterization
% 2. Simulate a TFP shock path (for large T)
% 3. Guess a solution path for the model (n-th solution path) corresponding 
% to n-th state vector path (and so prices).
% 4. Given n-th guess, solve backwards for expectation term in 
% intertemporal equation, filling in coutnerfactuals using ergodic theorem.
% 5. Given expectations, solve for optimal decision rules for all t.
% 6. Simulate forward, updating aggregate state vector ((n+1)-th path) and
% prices given optimal decision rules along with all other veriables.
% 7. Check convergence, MSE between (n+1)-th and n-th price vector. If not, 
% repeat steps 4-7.
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
%% Housekeeping 
clear;
close all;
clc;
addpath('./Functions');
addpath('./Figures');

%% Model fundamentals

% parameters 
%params.pFrisch = 1.000;
params.pEta = 6.400;
params.pRiskAversion = 1.000;
params.pBeta = 0.990;
params.pAlpha = 0.330;
params.pDelta = 0.025;
params.pMu = 0.600;
params.pRho = 0.950;
params.pSigmaEps = 0.009;
params.pNA = 7;

% TFP shock 
ss.A = 1;
[vA, mPA] = fnTauchen(params.pRho, params.pSigmaEps, 0.0, params.pNA);
vGridA = exp(vA);

% compute steady state 
ss.L = 1/3; % calibrate Frisch elasticity of labour supply to match 1/3 SS hours 
[ss.L, ss.K, ss.C, ss.Y, ss.I, ss.r, ss.w, params.pFrisch] = fnSSCompute(params,ss,1);

% unpack parameters 
pFrisch=params.pFrisch;pEta=params.pEta;pRiskAversion=params.pRiskAversion;pBeta=params.pBeta;pAlpha=params.pAlpha;pDelta=params.pDelta;pMu=params.pMu;

%% Repeated Transition Method (Nonlinear Global Solution in Sequence Space)

% time parameters 
BURNT = 500; % burn periods to ensure joint aggregte state (K,A) settles down (ergodic thm.)
Tmax = 3001;
T = Tmax + BURNT;
viT = (1:T)';
viFutureT = [(2:T)';T]; % assume T+1 state repeats state T (natural steady state terminal condition for the boundary)

% simulate shock (TFP) path 
initStateA = 4;
ivA = fnSimShock(mPA, T, initStateA, 1234);
vA = vGridA(ivA);
ivAFuture = ivA(viFutureT); % A shock at time t
vAFuture = vGridA(ivAFuture); % Aprime shock at time t 

% realised transition prob. vector
mPARealized = zeros(T,1);
for t = 1:T
    mPARealized(t) = mPA(ivA(t), ivAFuture(t));
end

% guess initial solution path
vK = ss.K .* ones(T, 1) + normrnd(0, 0.0001, T, 1); % perturbe K around SS for algorithm to work
vC = ss.C .* ones(T,1); % guess C initially flat (at SS) 
vL = ss.L .* ones(T,1); % (((1-pAlpha) .* vA .* vK.^pAlpha)./(pEta*vC.^pRiskAversion))^(pFrisch/(1+pFrisch*pAlpha);
vY = ss.Y .* ones(T,1); % vA.*vK.^(pAlpha).*vL.^(1-pAlpha);
vr = ss.r .* ones(T,1); % pAlpha.*vA.*(vK./vL).^(pAlpha-1) - pDelta;
vw = ss.w .* ones(T,1); % (1-pAlpha).*vA.*(vK./vL).^pAlpha;
vI = ss.I .* ones(T,1); % pDelta.*vK;

% vL = (((1-pAlpha).*ss.A.*vK.^pAlpha)./(pEta.*vC.^pRiskAversion)).^(pFrisch/(1+pFrisch*pAlpha));
% vY = ss.A.*vK.^(pAlpha).*vL.^(1-pAlpha);
% vr = pAlpha.*ss.A.*(vK./vL).^(pAlpha-1) - pDelta;
% vw = (1-pAlpha).*ss.A.*(vK./vL).^pAlpha;
% vI = pDelta.*vK;

% placeholder variables used for updating anchor variables 
vKnew = zeros(T,1); 
vCnew = ss.K.*ones(T,1);

% RTM loop parameters 
errTol = 1e-8;
wtOldK = 0.9000; 
wtOldC = 0.9000;
MaxIter = 10000;

% RTM loop 
iter = 1;
err = 10;
tic;
while err > errTol && iter <= MaxIter

    % ============================================================
    % BACKWARD SOLUTION FOR EXPECTATION TERM (BELIEFS)
    % ============================================================

    % future values for relevant aggregate state variables 
    vKprime = [vK(2:end);vK(1)];
    %vKprime = vK(viFutureT);
    %vKprime(end) = vK(1); % anchor beyond the simulated path K(T+1) at K(1) which will converge to SS-K 

    vV1 = 0; % reset expectaion term for current iteration 
    for iAprime = 1:params.pNA
        
        % find all (K,A) pairs in simulated path for counterfactual (unrealized) future state  
        Aprime = vGridA(iAprime); % unrealized future TFP shoc
        vCan = vK(find(ivA == iAprime));
        vCanLoc = find(ivA == iAprime);

        vCan(vCanLoc > T-BURNT) = [];
        vCan(vCanLoc < BURNT) = [];
        vCanLoc(vCanLoc > T-BURNT) = []; % eliminate all candidate locations (time periods) that fall inside burn period
        vCanLoc(vCanLoc < BURNT) = []; % eliminate all candidate locations (time periods) that fall inside burn period
        
        %vCanLoc = find(ivA == iAprime); % all time periods with counterfactual shock realized
        %vCanLoc(vCanLoc > T-BURNT) = []; % eliminate all candidate locations (time periods) that fall inside burn period
        %vCanLoc(vCanLoc < BURNT) = []; % eliminate all candidate locations (time periods) that fall inside burn period
        %vCan = vK(vCanLoc); % value of capital stock (K) at all counterfactual shock realizations
        [vCan, index] = sort(vCan); % sort K vector to interpolate Kprime over 
        vCanLoc = vCanLoc(index); % order candidate location according to sorted K index 
        
        % interpolate Kprime on plausible candidate locations 
        nLow = sum(repmat(vCan', length(vKprime), 1) < vKprime, 2); % identify all K candidates closest from below to Kprime
        nLow(nLow <= 1) = 1; % snap to lower limit index if Kprime is below all possible candidates
        nLow(nLow >= length(index)) = length(index) - 1; % snap to second largest index if Kprime is above all possible candidates 
        nHigh = nLow + 1; % position of closes K candidate from above to Kprime 
        wtLow = (vCan(nHigh) - vKprime) ./ (vCan(nHigh) - vCan(nLow)); % weight associated with closest K candidate from below to Kprime
        wtLow(wtLow>1) = 1; % snap lower weight to 1 if Kprime falls below possible candidates 
        wtLow(wtLow<0) = 0; % snap lower weight to 0 if Kprime falls above possible candidates 
        wtHigh = 1 - wtLow; % weight of associated with closest from above K candidate to Kprime 
        
        % compute value of controls in expectation term for all time t 
        % associated with future unrealized state vector (Kprime,Aprime)
        % value of controls are computed using linear interpolation 
        vCLow = vC(vCanLoc(nLow));
        vCHigh = vC(vCanLoc(nHigh));
        vLLow = (((1-pAlpha).*Aprime.*vKprime.^pAlpha) ./ (pEta.*vCLow.^pRiskAversion)).^(pFrisch/(1+pFrisch*pAlpha));
        vLHigh = (((1-pAlpha).*Aprime.*vKprime.^pAlpha) ./ (pEta.*vCHigh.^pRiskAversion)).^(pFrisch/(1+pFrisch*pAlpha));
        vrLow = pAlpha.*Aprime.*(vKprime./vLLow).^(pAlpha-1) - pDelta;
        vrHigh = pAlpha.*Aprime.*(vKprime./vLHigh).^(pAlpha-1) - pDelta;
        
        % cumulatively compute expectation term for counterfactuals (beliefs)
        vV1 = vV1 + ... 
            (ivAFuture ~= iAprime) .* ... % only add term if tomorrow's state is unrealised 
            pBeta .* mPA(ivA, iAprime) .* ...
            (wtLow.*(1+vrLow)./(vCLow.^pRiskAversion) + wtHigh.*(1+vrHigh)./(vCHigh.^pRiskAversion));    
    end
    
     % fill in realized component of expectation term (cumulatively)
     vLFuture = (((1-pAlpha).*vAFuture.*vKprime.^pAlpha) ./ (pEta.*vC(viFutureT).^pRiskAversion)).^(pFrisch/(1+pFrisch*pAlpha));
     vrFuture = pAlpha.*vAFuture.*(vKprime./vLFuture).^(pAlpha-1) - pDelta;
     vV1 = vV1 + ...
         pBeta .* mPARealized .* ...
         (1+vrFuture)./(vC(viFutureT).^pRiskAversion);
    
    % ============================================================
    % OPTIMAL BACKWARD SOLUTION FOR CONTROLS (FOCs & MCCs HOLD)
    % ============================================================

    % given expectation term (beliefs) solve forward for optimal
    % controls (endog. variables) path imposing FOCs and MCC. This
    % essentially updates the variable path for controls (not including 
    % endog. aggregate state path (K) or anchor variable(s) (C))
    vCfoc = (1./vV1).^(1/pRiskAversion);
    vL = (((1-pAlpha).*vA.*vK.^pAlpha) ./ (pEta.*vCfoc.^pRiskAversion)).^(pFrisch/(1+pFrisch*pAlpha));
    vr = pAlpha.*vA.*(vK./vL).^(pAlpha-1) - pDelta;
    vw = (1-pAlpha).*vA.*(vK./vL).^pAlpha;
    vY = vA.*vK.^pAlpha.*vL.^(1-pAlpha);
    vI = vY - vCfoc;
    
    % ============================================================
    % SIMULATE FORWARD
    % ============================================================

    % simulate forward endog. aggregate state (nonlinearly) using optimal
    % decision rule over controls 
    vKPast = [ss.K;vK(1:end-1)]; % lagged capital path anchored at SS-K
    vIPast = [ss.I;vI(1:end-1)]; % lagged investment path anchored at SS-I
    vKnew = (1-pDelta).*vKPast + vIPast;

    % new optimal (anchor) variable (C) given simulated/updated aggregate state path
    vCnew = vA.*vKnew.^pAlpha.*vL.^(1-pAlpha) - vI; % (L,I) path is optimal (satisfied FOCs and MCCs) for a converged aggregate state (K,A)

    % ============================================================
    % COMPUTE MSE FOR NEW UPDATED PATH
    % ============================================================
    
    MSE_C = mean((vC - vCnew).^2);
    MSE_K = mean((vK - vKnew).^2);
    err = mean([vC-vCnew; vK-vKnew].^2);
    
    % ============================================================
    % UPDATE (C,K) PATH
    % ============================================================
    
    vK = wtOldK.*vK + (1-wtOldK).*vKnew;
    vC = wtOldC.*vC + (1-wtOldC).*vCnew;

    % ============================================================
    % PROGRESS REPORTING 
    % ============================================================
    
    timer = toc;
    if mod(iter, 50)==0
        fprintf('Iteration %d. after %.2fs. MSE: %.10f\n', iter, timer, err);
        fprintf('MSE_C: %.6f. MSE_K: %.6f\n', MSE_C, MSE_K);
        fprintf('----------------------------------------\n')
        
        % plots 
        subplot(1,2,1);
        plot(1:T, vKnew(1:T), 'b-', 'LineWidth', 1.1);hold on;
        plot(1:T, vK(1:T), 'r-.', 'LineWidth', .9);
        yline(ss.K, 'k--', 'LineWidth', 1, 'Label', 'SS K');hold off;
        grid on;xlabel('Time');ylabel('K');xlim([1,T])
        legend('Actual', 'Predicted', '', 'Location', 'northeast')
        
        subplot(1,2,2);
        plot(1:T, vCnew(1:T), 'b-', 'LineWidth', 1.1);hold on;
        plot(1:T, vC(1:T), 'r-.', 'LineWidth', .9);
        yline(ss.C, 'k--', 'LineWidth', 1, 'Label', 'SS C');hold off;
        grid on;xlabel('Time');ylabel('C');xlim([1,T])
        legend('Actual', 'Predicted', '', 'Location', 'northeast')
        drawnow;pause(0.2);    
    end
    iter = iter + 1;
end
timer = timer/60;
if err <= errTol 
    fprintf('Model algorithm converged after %d iterations in %.2f mins!\n', iter, timer)
else
    fprintf('Model aglorithm failed to converge after %d iterations in %.2f mins!\n', iter, timer)
end

%% Figures 

set(gcf, 'Units', 'normalized');set(groot, 'defaultFigurePosition', [0.1, 0.1, 0.8, 0.8]);
set(gcf, 'Position', get(gcf, 'Position'));set(gcf, 'PaperPositionMode', 'auto')
saveas(gcf, './Figures/K_C_Path_NLGS_Fast.png')

% all variables time path
tStart = 1; tEnd=T;
figure;
subplot(3,2,1);
plot(vA,'b-','LineWidth',1);hold on;yline(ss.A,'k--');hold off;grid on;xlim([tStart,tEnd]);ylabel('A');xlabel('Time');
subplot(3,2,2);
plot(vY,'b-','LineWidth',1);hold on;yline(ss.Y,'k--');hold off;grid on;xlim([tStart,tEnd]);ylabel('Y');xlabel('Time');
subplot(3,2,3);
plot(vL,'b-','LineWidth',1);hold on;yline(ss.L,'k--');hold off;grid on;xlim([tStart,tEnd]);ylabel('L');xlabel('Time');
subplot(3,2,4);
plot(vI,'b-','LineWidth',1);hold on;yline(ss.I,'k--');hold off;grid on;xlim([tStart,tEnd]);ylabel('I');xlabel('Time');
subplot(3,2,5);
plot(vr,'b-','LineWidth',1);hold on;yline(ss.r,'k--');hold off;grid on;xlim([tStart,tEnd]);ylabel('r');xlabel('Time');
subplot(3,2,6);
plot(vw,'b-','LineWidth',1);hold on;yline(ss.w,'k--');hold off;grid on;xlim([tStart,tEnd]);ylabel('w');xlabel('Time');
set(gcf, 'Units', 'normalized');set(gcf, 'Position', [0.1, 0.1, 0.8, 0.8]);
saveas(gcf, './Figures/SolnPathGNLSFast.png')

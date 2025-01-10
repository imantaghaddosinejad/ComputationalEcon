%% Solving Canonical RBC Model Using Repeated Transition Method  %%%%%%%%%%
%
% 2025.01.10
% Author @ iman Taghaddosinejad (github.com/imantaghaddosinejad)
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% This file computes a RCE for a RBC model with enod. labor choice using a 
% nonlinear global solution method in sequence space. This version of the 
% code uses a loop over time periods which slows down computation speed in 
% exchange for clarity of the algorithm. For a fast vectorization approach
% see github.com/imantaghaddosinejad/ComputationalEcon/RBCFrischLabor/RBC_Frisch_GNLS_Fast.m.
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
addpath('./Functions')
addpath('./Figures')

%% Model Fundamentals

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

%% Repeated Transition Method (Nonlinear Global Solution in Sequence Space)

% time parameters 
BURNT = 500; % burn periods to ensure joint aggregte state (K,A) settles down (ergodic thm.)
Tmax = 2001;
T = Tmax + BURNT;

% simulate shock (TFP) path 
initStateA = 4;
ivA = fnSimShock(mPA, T, initStateA, 1234);
vA = vGridA(ivA);
% mTransActual = zeros(T,1); % vector containing prob. of actual realised A-path 
% for t = 1:T-1 
%     mTransActual(t,1) = mPA(ivA(t), ivA(t+1));
% end
% mTransActual(end, 1) = mPA(ivA(T), ivA(T)); % for final period assume same realised a for T+1 (natrual terinal SS boundary)

% guess path for variables (initital path)
vK = ss.K .* ones(T, 1) + normrnd(0, 0.000001, T, 1); % purturbe K around SS for algorithm to work
vC = ss.C .* ones(T,1); % guess C initially flat (at SS) 
vL = ss.L .* ones(T,1);%(((1/params.pEta) .* (1-params.pAlpha) .* vA .* vK.^params.pAlpha) ./ vC.^params.pRiskAversion).^(params.pFrisch/(1 + params.pFrisch*params.pAlpha));
vY = ss.Y .* ones(T,1);%vA .* vK.^(params.pAlpha) .* vL.^(1-params.pAlpha);
vr = ss.r .* ones(T,1);%params.pAlpha .* vA .* (vK./vL).^(params.pAlpha - 1) - params.pDelta;
vw = ss.w .* ones(T,1);%(1 - params.pAlpha) .* vA .* (vK./vL).^params.pAlpha;
vI = ss.I .* ones(T,1); %params.pDelta*ss.K .* ones(T,1);

% placeholder variables - anchor variables used for convergence criterion
vKnew = zeros(T,1); 
vCnew = ss.K .* ones(T,1);

% loop parameters 
errTol = 1e-8;
wtOldK = 0.9000;
wtOldC = 0.9000;
iter = 1;
MaxIter = 10000;

% RTM loop
err = 10;
tic;
while err > errTol && iter <= MaxIter

    % ============================================================
    % UPDATE VARIABLES PATH FOR TIME BETWEEN t=1 to t=T-1 
    % ============================================================
    
    tempV1 = zeros(T,1); % reset expectaion term vector for current iteration 
    for t = 1:T-1

        Kprime = vK(t+1); % future value of K at time t (realized)
        Cprime = vC(t+1); % future value of C at time t (realized)
        AprimeRealized = vA(t+1); % future value of A at time t (realized)

        % Fill in counterfactual terms in the expectation (RHS Euler Eq.). 
        % For every period t, find all (K(t),A) where A=A(t+1).
        % Find closest pair from below and above to realised capital, i.e.
        % K_l<= K(t+1) <= K_h where (K_l,K_h) are in the subset (K(t),A).
        % Interpolate K(t+1) on (K_l,K_h) and obtain corresponding weights
        % (wt_l,wt_h). Compute all relevant variables given state (K_l,A) and 
        % (K_h,A) that comprise the expectation term. Additively fill in the
        % counterfactual in the expectation term. The actual realised component 
        % of the expectation term will be filled in after this step. 
        for iAprime = 1:params.pNA

            Aprime = vGridA(iAprime); % potential future value of A at time t (unrealized)
            vCandidateLoc = find(vA == Aprime);
            vCandidate = vK(vCandidateLoc);            
            vCandidate(vCandidateLoc > T-BURNT) = []; % eliminate all candidates that fall inside BURNT period
            vCandidate(vCandidateLoc < BURNT) = []; % eliminate all candidates that fall inside BURNT period
            vCandidateLoc(vCandidateLoc > T-BURNT) = []; 
            vCandidateLoc(vCandidateLoc < BURNT) = [];
            [vCandidate, index] = sort(vCandidate); % sort candidates to find (K_l,K_h)
            vCandidateLoc = vCandidateLoc(index); % order candidate location according to sorted index
    
            % interpolate Kprime (K+1) on candidates 
            nLow = sum(vCandidate < Kprime);
            nLow(nLow<=1) = 1; % snap to lower limit if Kprime is possible candidates
            nLow(nLow >= length(index)) = length(index) - 1; % snap to second highest value if Kprime falls above all possible candidates
            nHigh = nLow + 1;
            wtLow = (vCandidate(nHigh) - Kprime) / (vCandidate(nHigh) - vCandidate(nLow)); % compute weight for how close Kprime is to upper candidate
            wtLow(wtLow>1) = 1; % snap lower weight to 1 if Kprime falls below possible candidates 
            wtLow(wtLow<0) = 0; % snap lower weight to 0 if Kprime falls above possible candidates 
            wtHigh = 1 - wtLow; 
            
            % compute counterfactual expectation term 
            % use Euler Equation and compute RHS which requires c(Kprime,Aprime)
            % and r(Kprime,Aprime). Use interpolation method and intratemporal
            % equation to deal with agg. labour supply to pin down r(Kprime,Aprime)
            LLow = (((1-params.pAlpha)*Aprime*Kprime^params.pAlpha)/(params.pEta*vC(vCandidateLoc(nLow))^params.pRiskAversion))^(params.pFrisch/(1+params.pFrisch*params.pAlpha));
            LHigh = (((1-params.pAlpha) * Aprime * Kprime^params.pAlpha) / (params.pEta * vC(vCandidateLoc(nHigh))^params.pRiskAversion))^(params.pFrisch/(1+params.pFrisch*params.pAlpha));
            rLow = (params.pAlpha * Aprime * (Kprime / LLow)^(params.pAlpha-1)) - params.pDelta;
            rHigh = (params.pAlpha * Aprime * (Kprime / LHigh)^(params.pAlpha-1)) - params.pDelta;
    
            % counterfactual expectation term (cumulatively computed sum term)
            tempV1(t) = tempV1(t) + ... % cumulatively add the counterfactual expectaiton terms 
                (ivA(t+1) ~= iAprime) * ... % only add term if tomorrow's state is unrealised 
                params.pBeta * mPA(ivA(t),iAprime) * ...
                (wtLow*(1+rLow)/vC(vCandidateLoc(nLow))^params.pRiskAversion + wtHigh*(1+rHigh)/vC(vCandidateLoc(nHigh))^params.pRiskAversion);
        end

        % fill in realized component of expectation term (cumulatively)
        Lprime = (((1-params.pAlpha)*AprimeRealized*Kprime^params.pAlpha)/(params.pEta*Cprime^params.pRiskAversion))^(params.pFrisch/(1+params.pFrisch*params.pAlpha));
        rprime = (params.pAlpha * AprimeRealized * (Kprime / Lprime)^(params.pAlpha-1)) - params.pDelta;
        tempV1(t) = tempV1(t) + ...
            params.pBeta * mPA(ivA(t),ivA(t+1)) * ...
            (1+rprime)/Cprime^params.pRiskAversion;

        % update all variables imposing Market Clearing and FOCs 
        Cfoc = 1/tempV1(t)^(1/params.pRiskAversion);
        vL(t) = (((1-params.pAlpha)*vA(t)*vK(t)^params.pAlpha) / (params.pEta*Cfoc^params.pRiskAversion))^(params.pFrisch/(1+params.pFrisch*params.pAlpha));
        vr(t) = (params.pAlpha * vA(t) * (vK(t)/vL(t))^(params.pAlpha-1)) - params.pDelta;
        vw(t) = (1-params.pAlpha) * vA(t) * (vK(t)/vL(t))^params.pAlpha;
        vY(t) = vA(t) * vK(t)^params.pAlpha * vL(t)^(1-params.pAlpha);
        vI(t) = vY(t) - Cfoc;

        % update state path and consumption path where the latter is done over the updated state path
        % compute new updated state value and consumption value given new state 
        if t == 1
            vKnew(t) = (1-params.pDelta)*ss.K + ss.I; % for t=1 use SS values always 
        else
            vKnew(t) = (1-params.pDelta)*vK(t-1) + vI(t-1); % this is a nonlinear capital accumulation path consistent with RCE
        end
        vCnew(t) = (vA(t) * vKnew(t)^params.pAlpha * vL(t)^(1-params.pAlpha)) - vI(t);
    end

    % ============================================================
    % UPDATE VARIABLES PATH FOR TIME t=T 
    % ============================================================

    % End point of simulated agg. state path
    % For period T we use (K(1),A(T)) as the future state. This assumes 
    % natural steady state terminal condition for the boundary. For large 
    % enough T with initial burn period, this is not an issue!
    % For all other variables, use period T value again inplace of T+1
    
    Kprime = vK(1); % set K(T+1) = K(1) which will converge to SS value (anchoring on SS)
    for iAprime = 1:params.pNA

            Aprime = vGridA(iAprime); % potential future value of A at time t (unrealized)
            vCandidateLoc = find(vA == Aprime);
            vCandidate = vK(vCandidateLoc);
            vCandidate(vCandidateLoc > T-BURNT) = []; % eliminate all candidates that fall inside BURNT period
            vCandidate(vCandidateLoc < BURNT) = []; % eliminate all candidates that fall inside BURNT period
            vCandidateLoc(vCandidateLoc > T-BURNT) = []; 
            vCandidateLoc(vCandidateLoc < BURNT) = [];
            [vCandidate, index] = sort(vCandidate); % sort candidates to find (K_l,K_h)
            vCandidateLoc = vCandidateLoc(index); % order candidate location according to sorted index
    
            % interpolate Kprime (K+1) on candidates 
            nLow = sum(vCandidate < Kprime);
            nLow(nLow<=1) = 1; % snap to lower limit if Kprime is possible candidates
            nLow(nLow >= length(index)) = length(index) - 1; % snap to second highest value if Kprime falls above all possible candidates
            nHigh = nLow + 1;
            wtLow = (vCandidate(nHigh) - Kprime) / (vCandidate(nHigh) - vCandidate(nLow)); % compute weight for how close Kprime is to upper candidate
            wtLow(wtLow>1) = 1; % snap lower weight to 1 if Kprime falls below possible candidates 
            wtLow(wtLow<0) = 0; % snap lower weight to 0 if Kprime falls above possible candidates 
            wtHigh = 1 - wtLow; 
            
            % compute counterfactual expectation term 
            % use Euler Equation and compute RHS which requires c(Kprime,Aprime)
            % and r(Kprime,Aprime). Use interpolation method and intratemporal
            % equation to deal with agg. labour supply to pin down r(Kprime,Aprime)
            LLow = (((1-params.pAlpha)*Aprime*Kprime^params.pAlpha)/(params.pEta*vC(vCandidateLoc(nLow))^params.pRiskAversion))^(params.pFrisch/(1+params.pFrisch*params.pAlpha));
            LHigh = (((1-params.pAlpha) * Aprime * Kprime^params.pAlpha) / (params.pEta * vC(vCandidateLoc(nHigh))^params.pRiskAversion))^(params.pFrisch/(1+params.pFrisch*params.pAlpha));
            rLow = (params.pAlpha * Aprime * (Kprime / LLow)^(params.pAlpha-1)) - params.pDelta;
            rHigh = (params.pAlpha * Aprime * (Kprime / LHigh)^(params.pAlpha-1)) - params.pDelta;
            
            % counterfactual expectation term (cumulatively computed sum term)
            tempV1(T) = tempV1(T) + ... % cumulatively add the counterfactual expectaiton terms 
                (ivA(T) ~= iAprime) * ... % only add term if tomorrow's state is unrealised 
                params.pBeta * mPA(ivA(T),iAprime) * ...
                (wtLow*(1+rLow)/vC(vCandidateLoc(nLow))^params.pRiskAversion + wtHigh*(1+rHigh)/vC(vCandidateLoc(nHigh))^params.pRiskAversion);

    end
    % realized component of expectation term 
    Cprime = vC(T); % reuse final period consumption level as state repeats in A for T+1 (but fix K(T+1) to K(1))
    Lprime = (((1-params.pAlpha)*vA(T)*Kprime^params.pAlpha)/(params.pEta*Cprime^params.pRiskAversion))^(params.pFrisch/(1+params.pFrisch*params.pAlpha));
    rprime = (params.pAlpha * AprimeRealized * (Kprime / Lprime)^(params.pAlpha-1)) - params.pDelta;
    tempV1(T) = tempV1(T) + ...
        params.pBeta * mPA(ivA(T), ivA(T)) * ... % prob. for realized transition 
        ( (1+rprime) / Cprime^params.pRiskAversion);

    % variable updating     
    Cfoc = 1/tempV1(T)^(1/params.pRiskAversion);
    vL(T) = (((1-params.pAlpha)*vA(T)*vK(T)^params.pAlpha) / (params.pEta*Cfoc^params.pRiskAversion))^(params.pFrisch/(1+params.pFrisch*params.pAlpha));
    vr(T) = (params.pAlpha * vA(T) * (vK(T)/vL(T))^(params.pAlpha-1)) - params.pDelta;
    vw(T) = (1-params.pAlpha) * vA(T) * (vK(T)/vL(T))^params.pAlpha;
    vY(T) = vA(T) * vK(T)^params.pAlpha * vL(T)^(1-params.pAlpha);
    vI(T) = vY(T) - Cfoc;
       
    vKnew(T) = (1-params.pDelta)*vK(T-1) + vI(T-1);
    vCnew(T) = (vA(T) * vKnew(T)^params.pAlpha * vL(T)^(1-params.pAlpha)) - vI(T);

    % ============================================================
    % COMPUTE MSE FOR NEW (C,K) PATH
    % ============================================================

    MSE_C = mean((vCnew - vC).^2);
    MSE_K = mean((vKnew - vK).^2);
    err = mean(([vCnew - vC; vKnew - vK]).^2);
    errK = vKnew - vK;

    % ============================================================
    % UPDATE (C,K) PATH
    % ============================================================

    vK = wtOldK.*vK + (1-wtOldK).*vKnew;
    vC = wtOldC.*vC + (1-wtOldC).*vCnew;

    % ============================================================
    % PROGRESS REPORTING
    % ============================================================

    timer = toc/60;
    
    minr = min(vr);
    maxr = max(vr);

    if mod(iter, 25) == 0
        fprintf('Iteration %d. after %.2f mins. MSE: %.10f\n', iter, timer, err);
        fprintf('MSE_C: %.6f. MSE_K: %.6f\n', MSE_C, MSE_K);
        fprintf('----------------------------------------\n')
        
        % plots 
        subplot(1,2,1);
        plot(1:T, vKnew(1:T), 'b-', 'LineWidth', 1.4);hold on;
        plot(1:T, vK(1:T), 'r-.', 'LineWidth', .9);
        yline(ss.K, 'k-', 'LineWidth', 1, 'Label', 'SS K');hold off;
        grid on;xlabel('Time');ylabel('K');xlim([1,T])
        legend('Actual', 'Predicted', '', 'Location', 'southeast')
        
        subplot(1,2,2);
        plot(1:T, vCnew(1:T), 'b-', 'LineWidth', 1.4);hold on;
        plot(1:T, vC(1:T), 'r-.', 'LineWidth', .9);
        yline(ss.C, 'k-', 'LineWidth', 1, 'Label', 'SS C');hold off;
        grid on;xlabel('Time');ylabel('C');xlim([1,T])
        legend('Actual', 'Predicted', '', 'Location', 'southeast')

        drawnow;
        pause(0.1);
    end
    iter = iter + 1;
end
if err <= errTol 
    fprintf('Model algorithm converged after %d iterations in %.2f mins!\n', iter, timer)
else
    fprintf('Model aglorithm failed to converge after %d iterations in %.2f mins!\n', iter, timer)
end

%% figures and post convergence analysis 

saveas(gcf, './Figures/K_path_C_path_GNLS_Slow.png')

% plot solution path 
tstart = 1;
tend = T;
figure;
subplot(3,2,1);
plot(tstart:tend, vA(tstart:tend), 'b-', 'LineWidth', 1.4);hold on;
yline(ss.A, 'k-', 'LineWidth', 1, 'Label','SS TFP'); hold off;
grid on;xlabel('Time');ylabel('TFP');xlim([tstart, tend]);

subplot(3,2,2);
plot(tstart:tend, vY(tstart:tend), 'b-', 'LineWidth', 1.4);hold on;
yline(ss.Y, 'k-', 'LineWidth', 1, 'Label','SS Y'); hold off;
grid on;xlabel('Time');ylabel('Y');xlim([tstart, tend]);

subplot(3,2,3);
plot(tstart:tend, vL(tstart:tend), 'b-', 'LineWidth', 1.4);hold on;
yline(ss.L, 'k-', 'LineWidth', 1, 'Label','SS L'); hold off;
grid on;xlabel('Time');ylabel('L');xlim([tstart, tend]);

subplot(3,2,4);
plot(tstart:tend, vI(tstart:tend), 'b-', 'LineWidth', 1.4);hold on;
yline(ss.I, 'k-', 'LineWidth', 1, 'Label','SS I'); hold off;
grid on;xlabel('Time');ylabel('I');xlim([tstart, tend]);

subplot(3,2,5);
plot(tstart:tend, vr(tstart:tend), 'b-', 'LineWidth', 1.4);hold on;
yline(ss.r, 'k-', 'LineWidth', 1, 'Label','SS r'); hold off;
grid on;xlabel('Time');ylabel('r');xlim([tstart, tend]);

subplot(3,2,6);
plot(tstart:tend, vw(tstart:tend), 'b-', 'LineWidth', 1.4);hold on;
yline(ss.w, 'k-', 'LineWidth', 1, 'Label','SS w'); hold off;
grid on;xlabel('Time');ylabel('w');xlim([tstart, tend]);
saveas(gcf, './Figures/SolnPath_GNLS_Slow.png')

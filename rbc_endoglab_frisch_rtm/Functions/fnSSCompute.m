%% Steady State Computation Canonical RBC Model %%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Args:
%   calib: (boolean) 1: to calibrate Frisch elasticity to match ss.L. 0:
%       specified Frisch elasticity parameter and compute ss.L
%   ss.L (scalar) SS labour supply if we're calibrating Frisch elasticity 
%   ssA: (scalar) steady state TFP 
%   pBeta: (scalar) discount factor 
%   pDelta: (scalar) depreciation rate of capital
%   pAlpha: (scalar) capital share of income (Cobb-Douglas)
%   pFrisch: (scalar) frisch elasticity of labour supply if we're not 
%       calibrating it 
%   pSigma: (scalar) coefficient of risk aversion
%   pEta: (scalar) disutility coefficient of labour supply
%
% Returns:
%   ssL: (scalar) steady state labour supply
%   ssK: (scalar) steady state capital supply
%   ssC: (scalar) steady state consumption
%   ssY: (scalar) steady state output 
%   ssI: (scalar) steady state investment 
%   ssr: (scalar) steady state interest rate 
%   ssw: (scalar) steady state wage rate 
%   pFrisch: (scalar) Frisch elasticity of labour supply 
%
function [ssL, ssK, ssC, ssY, ssI, ssr, ssw, pFrisch] = fnSSCompute(params, ss, calib)
    
    if nargin < 3
        calib = 0;
    end

    % unpack parameters 
    pBeta = params.pBeta;
    pDelta = params.pDelta;
    pAlpha = params.pAlpha;
    pSigma = params.pRiskAversion; 
    pEta = params.pEta; 
    ssA = ss.A;

    % SS per capita ratios - not dependent on Frisch elasticity directly 
    ssr = 1/pBeta - 1;
    R = ssr + pDelta;
    ssK2L = (R/(pAlpha*ssA))^(1 / (pAlpha - 1));
    ssw = (1-pAlpha) * ssA * (ssK2L)^pAlpha;

    if calib
    % compute SS values with predetermined ss.L = 1/3 and calibrate eta 

        % set SS labour supply - matching data 
        ssL = ss.L;

        % calibrate Frisch elasticity parameter term 
        pFrisch = 1 / (log(1/pEta * ssw * (ssA*ssK2L^pAlpha - pDelta*ssK2L)^(-pSigma))/log(ssL) - pSigma);
        
        % compute SS values 
        ssK = ssK2L * ssL;
        ssC = ssA * ssK^pAlpha * ssL^(1-pAlpha) - pDelta*ssK;
        ssY = ssA * ssK^pAlpha * ssL^(1-pAlpha);
        ssI = pDelta * ssK;

    else
    % compute SS values with predetermined eta parameter (pin ss.L)
        
        % set Frisch elasticity parameter 
        pFrisch = params.pFrisch;
        
        % SS labour supply
        ssL = (1/pEta * ssw * (ssA*ssK2L^pAlpha - pDelta*ssK2L)^(-pSigma))^(pFrisch/(1 + pFrisch*pSigma));
        
        % compute SS values 
        ssK = ssK2L * ssL;
        ssC = ssA * ssK^pAlpha * ssL^(1-pAlpha) - pDelta*ssK;
        ssY = ssA * ssK^pAlpha * ssL^(1-pAlpha);
        ssI = pDelta*ssK;
    end  
end

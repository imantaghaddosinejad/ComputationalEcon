%% SOLVING KRUSELL AND SMITH (1998) WITH ENDOGENOUS LABOUR SUPPLY %%%%%%%%% 
% 
% 2025.11.12
% Author @ Iman Taghaddosinejad (https://github.com/imantaghaddosinejad)
% 
% This file computes the staitonary equilibrium for Krusell and Smith
% (1998) using Policy Function Iteration + (linear) interpolation
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 
%========================================
% HOUSEKEEPING
%========================================
close all;
clc;
clear variables;
addpath('../Functions')
addpath('../Figures')
%%
%========================================
% MODEL FUNDAMENTALS 
%========================================
% model parameters 
p.alpha      = 0.36;    %  
p.beta       = 0.99;    % time-discount factor 
p.delta      = 0.025;   % capital depreciation rate 
p.frisch     = 1.00;    % Frisch elasticity of labour supply (this is fixed to 1.00 in computation)
p.eta        = 7.60;    % disutility of labour supply term 
p.rho        = 0.90;    % persistence in individual idyiosyncratic labour productivity shock
p.sigma      = 0.05;    % s.d. in individual idyiosyncratic labour productivity shock

% numerical parameters 
p.Nz = 7;           % number of idyiosyncratic productivity states 
p.Na = 100;         % number of points in (capital) asset grid 
p.agridmin = 0;     % minimum value of asset grid 
p.agridmax = 300;   % maximum value of asset grid (set large to not bind)
p.agridcurve = 7;   % degree of coarseness in asset grid (Maliar, Maliar, and Valli, 2010)

% wealth grid 
x = linspace(0,0.5,p.Na);
y = x.^p.agridcurve / max(x.^p.agridcurve);
vgrida = p.agridmin + (p.agridmax-p.agridmin).*y;

% individual productivity grid 
[vgridz, mtransz, ~] = fnTauchen(p.rho, p.sigma, 0.0, p.Nz);
vgridz = exp(vgridz);
clear x y % drop intermediary variables 
%%
%========================================
% NUMERICAL SOLUTION SETUP 
%========================================
% equilibrium objects 
mpolc = repmat(0.01.*vgrida', 1, p.Nz);
mpoln = zeros(size(mpolc));
mpolaprime_new = zeros(size(mpolc)); 
mlambda_new = zeros(size(mpolc));
Knew = 0;
Lnew = 0;
A = 1.0;

% auxillary objects 
mgrida = repmat(vgrida', 1, p.Nz); 
mgridz = repmat(vgridz', p.Na, 1);

% initial guess 
K = 36.5;
L = 0.33;
mpolaprime = zeros(size(mpolc));
mlambda = zeros(size(mpolc));
mcurrentdist = ones(p.Na,p.Nz)/(p.Na*p.Nz); % uniform initial distribution

% loop parameters 
maxiterge = 5000;
maxiterpfi = 10; % inexact inner solve caps PFI iterations to stabilise the outer GE fixed-point (avoids overshooting/non-contraction)
tol_pfi = 1e-4; % 
tol_ge = 1e-10;
tol_dist = 1e-10;
wt.w1 = 0.70000;
wt.w2 = 0.70000;
wt.w3 = 0.70000;
wt.w4 = 0.70000;
%% 
%========================================
% NUMERICAL SOLUTION 
%========================================
repge = true;
irepge = 50;
errge = 10;
iterge = 1;
timerge = tic; 
%====================
% 1. outer (ge) loop 
%====================
while errge > tol_ge && iterge <= maxiterge 

    % update prices 
    r = p.alpha * A * (K/L)^(p.alpha-1) - p.delta;
    w = (1-p.alpha) * A * (K/L)^p.alpha;

    %====================
    % 2. inner (pfi) loop
    %====================
    % low maxiterpfi: intentionally few PFI steps per GE iteration
    % behaves like a regularisation of the aggregate mapping and greatly improves GE stability
    % policies become accurate automatically as prices converge
    % view this as Howard-style improvement: few policy updates per GE step for stability
    errpfi = 10;
    iterpfi = 1;
    timerpfi = tic;
    while errpfi > tol_pfi && iterpfi <= maxiterpfi
    
        % compute beliefs (expectations)
        mexp = 0;
        for izp = 1:p.Nz         

            % future realised state  
            zprime = vgridz(izp);
            rprime = r;
            wprime = w;

            % optimal future policy rules 
            mpolaprimeprime = interp1(vgrida',squeeze(mpolaprime(:,izp)),mpolaprime,"linear","extrap"); % interpolate savings rule
            mprime = ((1+rprime)*mpolaprime - mpolaprimeprime)/(wprime*zprime);
            nprime = -mprime/2 + sqrt((p.eta*mprime).^2 + 4*p.eta)./(2*p.eta);
            cprime = (1+rprime)*mpolaprime + wprime*zprime*nprime - mpolaprimeprime;
            cprime(cprime<=0) = 1e-10; % safety lower bound for log-utility 
            muprime = 1./cprime;
            
            % update expectations
            mexp = mexp + repmat(mtransz(:,izp)',p.Na,1).*(1+rprime).*muprime;
        end

        % optimal intratemporal choices given beliefs 
        mexp = p.beta*mexp;
        c = 1./(mexp+mlambda); 
        n = w*mgridz./(p.eta*c);
        mlambda_new = 1./((1+r)*mgrida + w*mgridz.*n - mpolaprime) - mexp; % nnc computed as residual 
        mpolaprime_new = (1+r)*mgrida + w*mgridz.*n - c;

        % frictions (impose nnc) 
        mlambda_new(mpolaprime_new > p.agridmin) = 0;
        c = (c - (p.agridmin - mpolaprime_new)).*(mpolaprime_new<=p.agridmin) ...
            + c.*(mpolaprime_new>p.agridmin); % adjust consumption in instances where budget binds
        mpolaprime_new(mpolaprime_new<=p.agridmin) = p.agridmin;
        %c = mpolc;
        
        % compute error (pfi) 
        % errpfi = mean(abs([...
        %     sum(mpolaprime_new.*mcurrentdist,'all') - sum(mpolaprime.*mcurrentdist,'all');...
        %     sum(mlambda_new.*mcurrentdist,'all') - sum(mlambda.*mcurrentdist,'all')]),'all');
        
        errpfi = max(max(abs([mpolaprime_new-mpolaprime;mlambda_new-mlambda])));
        % updating 
        mpolaprime  = wt.w1*mpolaprime  + (1-wt.w1)*mpolaprime_new; 
        mlambda     = wt.w2*mlambda     + (1-wt.w2)*mlambda_new; 
        mpolc       = c; % jump update 
        mpoln       = n; % jump update 
        iterpfi = iterpfi+1;
    end
    timerpfi = toc(timerpfi);
    
    %====================
    % 3. stationary distribution (non-stochastic simulation)
    %====================
    errdist = 10;
    iterdist = 1;
    timerdist = tic;
    while errdist > tol_dist
        mnextdist = zeros(size(mcurrentdist)); % reset distribution to update 
        for iz = 1:p.Nz
            for ia = 1:p.Na
                aprime = mpolaprime(ia,iz);
                %aprime = mpolaprime_new(ia,iz);
                [lb,ub,wlb,wub] = fnInterp1dGrid(aprime,vgrida,p.Na); % interpolate policy on grid 
                mass = mcurrentdist(ia,iz); % transitioning mass over current state   
                for izp = 1:p.Nz
                    mnextdist(lb,izp) = mnextdist(lb,izp) + mass*mtransz(iz,izp)*wlb;
                    mnextdist(ub,izp) = mnextdist(ub,izp) + mass*mtransz(iz,izp)*wub;
                end
            end
        end
    
        % update distribution
        errdist = max(abs(mnextdist-mcurrentdist),[],"all"); % error 
        mcurrentdist = mnextdist;
        iterdist = iterdist+1;
    end
    timerdist = toc(timerdist); 
    
    %====================
    % 4. compute aggregates (mcc)
    %====================
    vmargdista = sum(mcurrentdist,2);
    Knew = vgrida*vmargdista; 
    Lnew = sum(mgridz.*mpoln.*mcurrentdist,'all');
    %mpolaprime_err = sum(mpolaprime_new.*mcurrentdist,'all') - sum(mpolaprime.*mcurrentdist,'all');
    %mlambda_err = sum(mlambda_new.*mcurrentdist,'all') - sum(mlambda.*mcurrentdist,'all');

    %====================
    % 5. updating (ge)
    %====================
    % error (ge) 
    errge = mean(abs([...
        Knew - K;...
        Lnew - L]),...
        'all');

    % update 
    K           = wt.w3*K           + (1-wt.w3)*Knew;
    L           = wt.w4*L           + (1-wt.w4)*Lnew;
    %mpolaprime  = wt.w1*mpolaprime  + (1-wt.w1)*mpolaprime_new; 
    %mlambda     = wt.w2*mlambda     + (1-wt.w2)*mlambda_new; 

    %====================
    % progress report
    %====================
    timerlapsed = toc(timerge);
    if repge == true && (mod(iterge, irepge) == 0 || iterge == 1 || errge <= tol_ge)
        % report 
        fprintf('---------------------------------- \n');
        fprintf('** Market Clearing Results ** \n'); 
        fprintf('iters_ge %d (%.2fs): err_ge = %.15f \n', iterge,timerlapsed,errge);
        fprintf('K = %.4f   L = %.4f \n',K,L);
        fprintf('r = %.4f   w = %.4f   max_lambda = %.4f \n',r,w,max(max(mlambda)));
        fprintf('** PFI Interim Report ** \n');
        fprintf('iters_pfi %d (%.2fs): err_pfi = %.15f \n',iterpfi,timerpfi,errpfi);
        fprintf('** Distribution Interim Report ** \n');
        fprintf('iters_dist %d (%.2fs): err_dist = %.15f \n',iterdist,timerdist,errdist);
        
        % save (mid)
        save('../Solutions/wip_ks1998endolabfrisch_ss.mat');
    end
    iterge = iterge+1;
end

% save (final)
%save('../Solutions/ks1998endolabfrisch_ss.mat');
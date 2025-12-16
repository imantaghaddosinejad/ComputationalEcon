%% SOLVING KRUSELL AND SMITH (1998) WITH ENDOGENOUS LABOUR SUPPLY %%%%%%%%% 
% 
% Author @ Iman Taghaddosinejad (https://github.com/imantaghaddosinejad)
% 2025.16.12
%
% This file computes the stationary equilibrium for Krusell and Smith
% (1998) using Policy Function Iteration + (linear) interpolation.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% HOUSEKEEPING %%

close all;
clc;
clear variables;
addpath('../Functions')
addpath('../Figures')

%=========================
% declare macros 
%=========================
repge = true; % interim progress report for ge loop
irepge = 100; % number of iterations per ge interim progress report 
distmethodeigenvc = 2; % distribution simulation method 

%% MODEL FUNDAMENTALS %%

%=========================
% model parameters 
%=========================
p.alpha      = 0.36;    % capital factor share of output 
p.beta       = 0.99;    % time-discount factor 
p.delta      = 0.025;   % capital depreciation rate 
p.frisch     = 1.00;    % Frisch elasticity of labour supply (this is fixed to 1.00 in computation)
p.eta        = 7.60;    % disutility of labour supply term 
p.rho        = 0.90;    % persistence in individual idiosyncratic labour productivity shock
p.sigma      = 0.05;    % s.d. in individual idiosyncratic labour productivity shock

%=========================
% numerical parameters 
%=========================
p.Nz = 7;           % number of idyiosyncratic productivity states 
p.Na = 100;         % number of points in (capital) asset grid 
p.agridmin = 0;     % minimum value of asset grid 
p.agridmax = 300;   % maximum value of asset grid
p.agridcurve = 7;   % degree of coarseness in asset grid (Maliar, Maliar, and Valli, 2010)

%=========================
% wealth grid 
%=========================
x = linspace(0,0.5,p.Na);
y = x.^p.agridcurve / max(x.^p.agridcurve);
vgrida = p.agridmin + (p.agridmax-p.agridmin).*y;
clear x y % drop intermediary variables 

%=========================
% individual productivity grid 
%=========================
[vgridz, mtransz, ~] = fnTauchen(p.rho, p.sigma, 0.0, p.Nz);
vgridz = exp(vgridz);

%% NUMERICAL SETUP %%

%=========================
% equilibrium objects 
%=========================
mpolc = repmat(0.01.*vgrida', 1, p.Nz);
mpoln = zeros(size(mpolc));
mpolaprime_new = zeros(size(mpolc)); 
mlambda_new = zeros(size(mpolc));
Knew = 0;
Lnew = 0; 
A = 1.0; % tfp

%=========================
% auxillary objects 
%=========================
mgrida = repmat(vgrida', 1, p.Nz); 
mgridz = repmat(vgridz', p.Na, 1);

%=========================
% initial guess 
%=========================
K = 36.5;
L = 0.33;
mpolaprime = zeros(size(mpolc));
mlambda = zeros(size(mpolc));
mcurrentdist = ones(p.Na,p.Nz)/(p.Na*p.Nz); % uniform initial distribution

%=========================
% loop parameters 
%=========================
%maxiterge = 5000; % limit max number of GE iterations
tol_ge = 1e-10;
tol_dist = 1e-10;
wt.w1 = 0.80000;
wt.w2 = 0.80000;
wt.w3 = 0.80000;
wt.w4 = 0.80000;

%% NUMERICAL SOLUTION %%

%=========================
% continue from last save point
%=========================
% load('../Solutions/wip_ks1998endolabfrisch_ss.mat')

%=========================
% 1. outer (ge) loop 
%=========================
errge = 10;
iterge = 1;
timerge = tic; 
while errge > tol_ge %&& iterge <= maxiterge 

    % update prices 
    r = p.alpha * A * (K/L)^(p.alpha-1) - p.delta;
    w = (1-p.alpha) * A * (K/L)^p.alpha;

    %=========================
    % 2. pfi step
    %=========================
    % simultaneously update policy functions with prices to boost speed and
    % improve GE convergence stability. Within each GE step do one PFI
    % (view this as a Howard-style improvement). 
    
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
        
    % updating 
    mpolc       = c; % jump update 
    mpoln       = n; % jump update 
    
    %=========================
    % 3. stationary distribution (non-stochastic simulation)
    %=========================
    timerdist = tic;

    % option 1: histogram method
    if distmethodeigenvc == 1 
        errdist = 10;
        iterdist = 1;
        while errdist > tol_dist
            mnextdist = zeros(size(mcurrentdist)); % reset distribution to update 
            for iz = 1:p.Nz
                for ia = 1:p.Na
                    aprime = mpolaprime_new(ia,iz);
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
        end
        
        % option 2: eigenvalue method 
    elseif distmethodeigenvc == 2
        I = zeros(p.Nz*p.Na*2*p.Nz,1); % row indices (each state (ia,iz) has 2*Nz transitions due to interpolation)
        J = zeros(p.Nz*p.Na*2*p.Nz,1); % column indices 
        V = zeros(p.Nz*p.Na*2*p.Nz,1); % values (weighted transition prob)
        ctr = 0; % reset transition-record index (purely bookkeeping) 
        for iz = 1:p.Nz
            for ia = 1:p.Na
                icurr = sub2ind([p.Na,p.Nz],ia,iz); % current state index 
                aprime = mpolaprime_new(ia,iz);
                [lb,ub,wlb,wub] = fnInterp1dGrid(aprime,vgrida,p.Na); % interpolate policy on grid
                for izp = 1:p.Nz
                    ctr = ctr + 1; % first transition given izp (lb-case)
                    I(ctr) = icurr;
                    J(ctr) = sub2ind([p.Na,p.Nz],lb,izp); % future state index lb-case 
                    V(ctr) = mtransz(iz,izp)*wlb;
                    
                    ctr = ctr + 1; % second transition given izp (ub-case)
                    I(ctr) = icurr;
                    J(ctr) = sub2ind([p.Na,p.Nz],ub,izp); % future state index ub-case 
                    V(ctr) = mtransz(iz,izp)*wub;                
                end
            end
        end
        mtransjoint = sparse(I,J,V,p.Na*p.Nz,p.Na*p.Nz); % declare sparse joint-transition matrix to efficiently handle zero entries 
        
        % invariant distribution (left eigenvector)
        [eigvc,~] = eigs(mtransjoint',1);
        eigvc = real(eigvc);
        eigvc = eigvc / sum(eigvc); % normalise distribution 
        eigvc(eigvc<0) = 0; % enforce valid probability lower bound (avoid numerical negatives)
        eigvc = eigvc / sum(eigvc); % re-normalize 
        mcurrentdist = reshape(eigvc,[p.Na,p.Nz]); % reshape into 2D distribution
    end
    timerdist = toc(timerdist); 
    
    %=========================
    % 4. compute aggregates (mcc)
    %=========================
    vmargdista = sum(mcurrentdist,2);
    Knew = vgrida*vmargdista; 
    Lnew = sum(mgridz.*mpoln.*mcurrentdist,'all');

    %=========================
    % 5. updating (ge)
    %=========================
    % error (ge) 
    mpolaprime_err = sum(mpolaprime_new.*mcurrentdist,'all') - sum(mpolaprime.*mcurrentdist,'all');
    mlambda_err = sum(mlambda_new.*mcurrentdist,'all') - sum(mlambda.*mcurrentdist,'all');
    errge = mean(abs([...
        Knew - K;...
        Lnew - L;...
        mlambda_err]),...
        'all');

    % updating (damped/convex)
    mpolaprime  = wt.w1*mpolaprime  + (1-wt.w1)*mpolaprime_new; 
    mlambda     = wt.w2*mlambda     + (1-wt.w2)*mlambda_new; 
    K           = wt.w3*K           + (1-wt.w3)*Knew;
    L           = wt.w4*L           + (1-wt.w4)*Lnew;

    %=========================
    % progress report
    %=========================
    timerlapsed = toc(timerge);
    if repge == true && (mod(iterge, irepge) == 0 || iterge == 1 || errge <= tol_ge)
        % report 
        fprintf('---------------------------------- \n');
        fprintf('** Market Clearing Results ** \n'); 
        fprintf('iters_ge %d (%.2fs): err_ge = %.15f \n',iterge,timerlapsed,errge);
        fprintf('K = %.4f   L = %.4f \n',K,L);
        fprintf('r = %.4f   w = %.4f   max_lambda = %.4f \n',r,w,max(max(mlambda)));
        fprintf('mpolaprime_err = %.15f \n',abs(mpolaprime_err));
        if distmethodeigenvc == 1 
            fprintf('** Distribution Interim Report ** \n');
            fprintf('iters_dist %d (%.2fs): err_dist = %.15f \n',iterdist,timerdist,errdist);
        end

        % plot 
        %plot(vgrida,vmargdista,'LineWidth',1.5);grid on;xlim([0 100]);drawnow;
        %pause(0.1);

        % save (mid)
        save('../Solutions/wip_ks1998endolabfrisch_ss.mat');
    end
    iterge = iterge+1;
end

%=========================
% save (final)
%=========================
save('../Solutions/ks1998endolabfrisch_ss.mat');

%%
%=========================
% solution report 
%=========================
load('../Solutions/ks1998endolabfrisch_ss.mat') 

% wealth (marginal) distribution plot 
figure;
plot(vgrida,vmargdista,'LineWidth',1.5);grid on;xlim([0,100]);
xlabel('Assets','FontSize',14);
ylabel('Density','FontSize',14);
ax = gca;
ax.FontSize = 15; 
set(gcf, 'Units', 'inches', 'Position', [1 1 5 4]);
exportgraphics(gcf, '../Figures/wealth_dist_ss.pdf', 'ContentType', 'vector');

% wealth-by-productivity state distribution plot 
figure;
plot(vgrida,mcurrentdist,'LineWidth',1.5);grid on;xlim([0,100]);
xlabel('Assets','FontSize',14);
ylabel('Density','FontSize',14)
legend('z1','z2','z3','z4','z5','z6','z7','Location','northeast','FontSize',14);
ax = gca;
ax.FontSize = 15; 
set(gcf, 'Units', 'inches', 'Position', [1 1 5 4]);
exportgraphics(gcf, '../Figures/wealth_by_prod_dist_ss.pdf', 'ContentType', 'vector');

% savings policy rule
figure;
plot(vgrida,mpolaprime(:,1),'LineWidth',1.2);hold on;
plot(vgrida,mpolaprime(:,end),'LineWidth',1.2);
hold off;
grid on;
ylabel('Hours');
xlabel('Assets');
legend('Lowest z','Highest z','Location','southeast');
ax = gca;
ax.FontSize = 15; 
set(gcf, 'Units', 'inches', 'Position', [1 1 5 4]);
exportgraphics(gcf, '../Figures/savings_policy.pdf', 'ContentType', 'vector');

% labour supply policy rule
figure;
plot(vgrida,mpoln(:,1),'LineWidth',1.2);hold on;
plot(vgrida,mpoln(:,end),'LineWidth',1.2);
hold off;
grid on;
ylabel('Hours');
xlabel('Assets');
legend('Lowest z','Highest z','Location','northeast');
ax = gca;
ax.FontSize = 15; 
set(gcf, 'Units', 'inches', 'Position', [1 1 5 4]);
exportgraphics(gcf, '../Figures/hours_policy.pdf', 'ContentType', 'vector');

% consumption policy rule
figure;
plot(vgrida,mpolc(:,1),'LineWidth',1.2);hold on;
plot(vgrida,mpolc(:,end),'LineWidth',1.2);
hold off;
grid on;
ylabel('Consumption');
xlabel('Assets');
legend('Lowest z','Highest z','Location','southeast');
ax = gca;
ax.FontSize = 15; 
set(gcf, 'Units', 'inches', 'Position', [1 1 5 4]);
exportgraphics(gcf, '../Figures/consumption_policy.pdf', 'ContentType', 'vector');
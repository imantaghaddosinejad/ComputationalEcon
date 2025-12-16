%% SOLVING KRUSELL AND SMITH (1998) WITH ENDOGENOUS LABOUR SUPPLY %%%%%%%%%
%
% 2025.16.12
% Author @ Iman Taghaddosinejad (https://github.com/imantaghaddosinejad)
%
% This file computes the Recursive Competitive Equilibrium (RCE) for
% Krusell and Smith (1998) using the Repeated Transition Method (RTM)
% developed by Lee (2025). The RTM provides a global nonlinear solution.
%
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
irepge = 20; % number of iterations per ge interim progress report 

%% MODEL FUNDAMENTALS %%

% load steady-sate solution 
ss = load('../Solutions/ks1998endolabfrisch_ss.mat');

%=========================
% unpack key steady-state objects 
%=========================
p = ss.p;
vgrida = ss.vgrida;
vgridz = ss.vgridz;
mtransz = ss.mtransz;

%=========================
% aggregate shock 
%=========================
p.NA = 2;
vgridA = [0.99,1.01];
mtransA = [0.875,0.125;0.125,0.875];

%% NUMERICAL SETUP %%

%=========================
% set transition path
%=========================
Tlen = 4000; % total length of transition path (+1 to anchor t=1 to steady-state)
burnin = 500; % burnin period length to for ergodicity 
T = Tlen + burnin;
itransp = [(2:T)';T]; % set t=T+1 off-the-transition path to repeat final period (anchor on t=T)
win_t = ((burnin+1):(T-burnin))'; % compute ge errors over non-burnin periods along transition path

%=========================
% simulate aggregate shock
%=========================
p.initA = 1; 
seed = 100;
rng(seed);
ishock_t = fnSimShock(mtransA,T,p.initA);
ishock_tp = ishock_t(itransp); % shock index t+1 at time t 

%=========================
% declare equilibrium objects
%=========================
mpolc_t             = zeros(p.Na,p.Nz,T);   % consumption policy rule 
mpoln_t             = zeros(size(mpolc_t)); % labour supply policy rule 
mpolaprime_t        = zeros(size(mpolc_t)); % savings policy rule 
mlambda_t           = zeros(size(mpolc_t)); % nnc 
mpolaprime_new_t    = zeros(size(mpolc_t));
mlambda_new_t       = zeros(size(mpolc_t));
C_t                 = zeros(T,1);           % aggregate consumption 

%=========================
% start and end points 
%=========================
startdist = ss.mcurrentdist; % anchor t=1 to steady-state 
for t = 1:T
    mpolc_t(:,:,t) = ss.mpolc; % anchor t=T+1 to t=T for policy rules
    mpoln_t(:,:,t) = ss.mpoln;
end

%=========================
% initial guess 
%=========================
K_t = ones(T+1,1) * ss.K + normrnd(0,0.0001,T+1,1); % anchor t=T+1 to steady-state
L_t = ones(T,1) * ss.L;
for t = 1:T 
    mpolaprime_t(:,:,t) = ss.mpolaprime;
    mlambda_t(:,:,t) = ss.mlambda;
end
polaprime_t = sum(ss.mpolaprime.*ss.mcurrentdist,'all').*ones(T,1); % average savings over transition path
lambda_t = sum(ss.mlambda.*ss.mcurrentdist,'all').*ones(T,1); % average nnc over transition path

%=========================
% numerical parameters 
%=========================
tol_ge = 1e-6;
wt.w1 = 0.9000;
wt.w2 = 0.9000;
wt.w3 = 0.9000;
wt.w4 = 0.9000;

%% NUMERICAL SOLUTION %% 

%=========================
% continue from last save point
%=========================
%load('../Solutions/wip_ks1998endolabfrisch_bs.mat');

%=========================
% 1. outer (ge) loop
%=========================
iter_ge = 1;
err_ge = 10;
timer_ge = tic;
while err_ge > tol_ge 
            
    %=========================
    % 2. backward solution
    %=========================
    for t = T:-1:1

        % current aggregate state 
        iA = ishock_t(t);
        A = vgridA(iA);
        K = K_t(t);
        L = L_t(t);

        % current prices (uniquely pinned given aggregates)
        r = p.alpha*A*(K/L)^(p.alpha-1)-p.delta;
        w = (1-p.alpha)*A*(K/L)^p.alpha;

        % future aggregate state (for beliefs) 
        ifuture = itransp(t); % future time-index 
        ifutureshock = ishock_t(ifuture); % shock 
        Kprime = K_t(t+1); % capital 

        % expected future value (rational expectations) 
        mexp = 0;
        for iAprime = 1:p.NA 
            Aprime = vgridA(iAprime); % future shock value 
            
            for izprime = 1:p.Nz
                zprime = vgridz(izprime); % individual/idiosyncratic future productivity shock
                
                % compute counterfactual expectations for unrealised aggregate states
                if ifutureshock ~= iAprime || t == T
                    % find a period where future shock realisation is same
                    % as Aprime and Kprime is closest from below and above
                    can = K_t(find(ishock_t==iAprime)); % candidates with counterfactual exog. aggregate state 
                    canloc = find(ishock_t==iAprime); % location of candidates 
                    can(canloc>T-burnin) = []; % last burnin periods cannot be candidates 
                    can(canloc<burnin) = []; % initial burnin periods cannot be candidates
                    canloc(canloc>T-burnin) = [];
                    canloc(canloc<burnin) = []; 
                    [can,index] = sort(can); % sort candidates to find closest from below and above
                    canloc = canloc(index); % save sorted locations 
                    
                    % find period where capital stock is closest
                    Klow = sum(can<Kprime); % period capital stock is closest to Kprime from below 
                    Klow(Klow<=1) = 1; % snap to first period if location is below candidate set 
                    Klow(Klow>=length(index)) = length(index)-1; % snap to second to last period if locaiton is above candidate set
                    Khigh = Klow+1; % period capital stock is closest to Kprime from above
                    weightlow = (can(Khigh)-Kprime)/(can(Khigh)-can(Klow)); % linear interpolation weight on lower side
                    weightlow(weightlow<0) = 0; % snap to upper bound if Kprime is above candidate set 
                    weightlow(weightlow>1) = 1; % snap to lower bound if Kprime is below candidate set
                    
                    % interpolate aggregates in counterfactual future state
                    K2Lprime_low = Kprime/L_t(canloc(Klow));
                    rprime_low = p.alpha*Aprime*(K2Lprime_low)^(p.alpha-1)-p.delta;
                    wprime_low = (1-p.alpha)*Aprime*(K2Lprime_low)^p.alpha;
                    
                    K2Lprime_high = Kprime/L_t(canloc(Khigh));
                    rprime_high = p.alpha*Aprime*(K2Lprime_high)^(p.alpha-1)-p.delta;
                    wprime_high = (1-p.alpha)*Aprime*(K2Lprime_high)^p.alpha;

                    % interpolate savings in counterfactual future state
                    mpolaprimeprime_low = interp1(vgrida',squeeze(mpolaprime_t(:,izprime,canloc(Klow))),...
                        squeeze(mpolaprime_t(:,:,t)),"linear","extrap");
                    mpolaprimeprime_high = interp1(vgrida',squeeze(mpolaprime_t(:,izprime,canloc(Khigh))),...
                        squeeze(mpolaprime_t(:,:,t)),"linear","extrap");
                    
                    % interpolate consumption in counterfactual future state
                    mprime_low = ((1+rprime_low)*squeeze(mpolaprime_t(:,:,t)) - mpolaprimeprime_low)./(wprime_low*zprime); % auxillary object
                    nprime_low = -mprime_low/2 + sqrt((p.eta*mprime_low).^2 + 4*p.eta)/(2*p.eta); % future labour supply
                    cprime_low = (1+rprime_low)*squeeze(mpolaprime_t(:,:,t)) + wprime_low*zprime*nprime_low - mpolaprimeprime_low; % future consumption
                    cprime_low(cprime_low<=0) = 1e-10; % lower bound for log-utility 

                    mprime_high = ((1+rprime_high)*squeeze(mpolaprime_t(:,:,t)) - mpolaprimeprime_high)./(wprime_high*zprime); % auxillary object
                    nprime_high = -mprime_high/2 + sqrt((p.eta*mprime_high).^2 + 4*p.eta)/(2*p.eta); % future labour supply
                    cprime_high = (1+rprime_high)*squeeze(mpolaprime_t(:,:,t)) + wprime_high*zprime*nprime_high - mpolaprimeprime_high; % future consumption
                    cprime_high(cprime_high<=0) = 1e-10; % lower bound for log-utility
            
                    % expectations (RHS of Euler) for counterfactual future state
                    muprime_low = 1./cprime_low;
                    muprime_high = 1./cprime_high;
                    mexp = mexp ...
                        + weightlow*((1+rprime_low)*muprime_low.*repmat(mtransz(:,izprime)',p.Na,1)*mtransA(iA,iAprime)) ...
                        + (1-weightlow)*((1+rprime_high)*muprime_high.*repmat(mtransz(:,izprime)',p.Na,1)*mtransA(iA,iAprime));

                % compute on the path expectation for realised future aggregate state  
                else
                    % aggregates in realised future state
                    K2Lprime = Kprime/L_t(ifuture);
                    rprime = p.alpha*Aprime*(K2Lprime)^(p.alpha-1)-p.delta;
                    wprime = (1-p.alpha)*Aprime*(K2Lprime)^p.alpha;

                    % interpolate savings in realised future state
                    mpolaprimeprime = interp1(vgrida',squeeze(mpolaprime_t(:,izprime,ifuture)),...
                        squeeze(mpolaprime_t(:,:,t)),"linear","extrap");

                    % consumption in realised future state 
                    mprime = ((1+rprime)*squeeze(mpolaprime_t(:,:,t)) - mpolaprimeprime)./(wprime*zprime); % auxillary object
                    nprime = -mprime/2 + sqrt((p.eta*mprime).^2 + 4*p.eta)./(2*p.eta); % future labour supply 
                    cprime = (1+rprime)*squeeze(mpolaprime_t(:,:,t)) + wprime*zprime*nprime - mpolaprimeprime; % future consumption
                    cprime(cprime<=0) = 1e-10; % lower bound for log-utility
                    muprime = 1./cprime;

                    % expectations (RHS of Euler) for realised future state
                    mexp = mexp ...
                        + (1+rprime)*muprime.*repmat(mtransz(:,izprime)',p.Na,1)*mtransA(iA,ifutureshock);
                end
            end
        end

        % given beliefs compute intratemporal optimal choices
        mexp = p.beta*mexp; 
        c = 1./(mexp + mlambda_t(:,:,t)); % consumption 
        n = (w*repmat(vgridz',p.Na,1))./(p.eta*c); % labour supply 
        mlambda_new_temp = 1./((1+r)*repmat(vgrida',1,p.Nz) + w*repmat(vgridz',p.Na,1).*n - squeeze(mpolaprime_t(:,:,t))) - mexp; % nnc as residual
        mpolaprime_new_temp = (1+r)*repmat(vgrida',1,p.Nz) + w*repmat(vgridz',p.Na,1).*n - c; % savings

        % frictions (enforce borrowing constraint)
        mlambda_new_temp(mpolaprime_new_temp>p.agridmin) = 0;
        c = (c - (p.agridmin-mpolaprime_new_temp)).*(mpolaprime_new_temp<=p.agridmin) ...
            + c.*(mpolaprime_new_temp>p.agridmin);
        mpolaprime_new_temp(mpolaprime_new_temp<=p.agridmin) = p.agridmin;

        % update 
        mlambda_new_t(:,:,t) = mlambda_new_temp;
        mpolaprime_new_t(:,:,t) = mpolaprime_new_temp;
        mpolc_t(:,:,t) = c; % jump update 
        %mpolc_t(:,:,t) = wt.w4*mpolc_t(:,:,t) + (1-wt.w4)*c; % conservative damped updating
        mpoln_t(:,:,t) = n; % jump update
    end

    %=========================
    % 3. non-stochastic smulation
    %=========================
    % reset placeholders 
    Knew_t = zeros(size(K_t)); 
    Lnew_t = zeros(T,1);
    lambda_t = zeros(T,1);
    lambda_new_t = zeros(T,1);
    polaprime_t = zeros(T,1);
    polaprime_new_t = zeros(T,1);
    
    % initial distribution
    currentdist = startdist; % initial distribution is anchored to steady-state
    Knew_t(1) = vgrida*sum(currentdist,2); % initial aggregate K is anchored to steady-state
    
    % simulate forward 
    for t = 1:T
    
        iA = ishock_t(t); % current period shock state
        nextdist = zeros(size(currentdist)); % reset next period distribution

        for iz = 1:p.Nz 
            for ia = 1:p.Na 
                aprime = mpolaprime_new_t(ia,iz,t);
                [lb,ub,wtlb,wtub] = fnInterp1dGrid(aprime,vgrida,p.Na);
                mass = currentdist(ia,iz);

                for izprime = 1:p.Nz                    
                    nextdist(lb,izprime) = nextdist(lb,izprime) ...
                        + wtlb*mtransz(iz,izprime)*mass;
                    nextdist(ub,izprime) = nextdist(ub,izprime) ...
                        + wtub*mtransz(iz,izprime)*mass;
                end
            end
        end

        % update 
        Knew_t(t+1) = vgrida*sum(nextdist,2); % capital mcc 
        Lnew_t(t) = sum(repmat(vgridz',p.Na,1).*mpoln_t(:,:,t).*currentdist,'all'); % labour supply mcc 
        C_t(t) = sum(squeeze(mpolc_t(:,:,t)).*currentdist,'all'); % consumption
        lambda_t(t) = sum(squeeze(mlambda_t(:,:,t)).*currentdist,'all'); % average nnc given last iteration policy rules
        lambda_new_t(t) = sum(squeeze(mlambda_new_t(:,:,t)).*currentdist,'all'); % average nnc given current iteration policy rules
        polaprime_t(t) = sum(squeeze(mpolaprime_t(:,:,t)).*currentdist,'all'); % average savings given last iteration policy rules
        polaprime_new_t(t) = sum(squeeze(mpolaprime_new_t(:,:,t)).*currentdist,'all'); % average savings given current iteration policy rules
        currentdist = nextdist; % update transitioning distribution into next period 
    end

    %=========================
    % 4. check convergence and update prices
    %=========================
    % market clearing error (mse) 
    err_ge = mean(abs([...
        Knew_t(win_t) - K_t(win_t);...
        Lnew_t(win_t) - L_t(win_t);...
        lambda_new_t(win_t) - lambda_t(win_t)].^2),'all');
    err_K = Knew_t-K_t;
    err_L = Lnew_t-L_t;
    %rmse_ge = sqrt(mean(abs([...
    %    Knew_t(win_t) - K_t(win_t);...
    %    Lnew_t(win_t) - L_t(win_t);...
    %    lambda_new_t(win_t) - lambda_t(win_t)].^2),'all'));


    % update 
    K_t             = wt.w1*K_t             + (1-wt.w1)*Knew_t;
    L_t             = wt.w2*L_t             + (1-wt.w2)*Lnew_t;
    mlambda_t       = wt.w3*mlambda_t       + (1-wt.w3)*mlambda_new_t;
    mpolaprime_t    = wt.w4*mpolaprime_t    + (1-wt.w4)*mpolaprime_new_t;

    %=========================
    % progress report
    %=========================
    timerlapsed = toc(timer_ge);
    if repge == true && (mod(iter_ge, irepge) == 0 || iter_ge == 1 || err_ge <= tol_ge)
        % report 
        fprintf('\n ------------------------------------ \n');
        fprintf('** Market Clearing Results ** \n');
        fprintf('iter_ge %d (%.3fs): err_ge = %.10f \n',iter_ge,timerlapsed,err_ge);
        %fprintf('err_rmse_ge = %.10f \n', rmse_ge);
        fprintf('err_max_K = %.10f   err_max_L = %.10f \n',max(abs(err_K)),max(abs(err_L)));
        

        % plot
        subplot(1,2,1);
        plot(1:T,K_t(1:T),'b-','LineWidth',1);hold on;
        plot(1:T,Knew_t(1:T),'r-.','LineWidth',1);hold off;
        grid on;
        ylabel('K','FontSize',14);
        yline(ss.K,'k--','LineWidth',1);
        xlim([1,T]);
        legend('Predicted','Realized','','Location','northeast');
        
        subplot(1,2,2);
        plot(1:T,L_t(1:T),'b-','LineWidth',1);hold on;
        plot(1:T,Lnew_t(1:T),'r-.','LineWidth',1);hold off;
        grid on;
        ylabel('L','FontSize',14);
        yline(ss.L,'k--','LineWidth',1);
        xlim([1,T]);
        legend('Predicted','Realized','','Location','northeast');
        drawnow;        
        %pause(0.1);

        % save (mid)
        % save('../Solutions/wip_ks1998endolabfrisch_bs.mat');
    end
    iter_ge = iter_ge+1;
end

%=========================
% compute aggregates and prices 
%=========================
Y_t = vgridA(ishock_t)'.*K_t(1:T).^p.alpha.*L_t.^(1-p.alpha); % output 
I_t = K_t([(2:T)';T]) - (1-p.delta)*K_t([1;(1:T-1)']); % investment 
Ctemp_t = Y_t - I_t; % alternative computation of aggregate consumption 
Ygap_t = C_t - Ctemp_t; % output gap (goods market clearing) 
r_t = p.alpha*vgridA(ishock_t)'.*(K_t(1:T)./L_t).^(p.alpha-1)-p.delta;
w_t = (1-p.alpha)*vgridA(ishock_t)'.*(K_t(1:T)./L_t).^p.alpha;

%=========================
% save (final)
%=========================
% save('../Solutions/ks1998endolabfrisch_bs.mat');
%%
%=========================
% solution report
%=========================
load('../Solutions/ks1998endolabfrisch_bs.mat');

% HP-filter log series
[~,Y_hp]    = hpfilter(log(Y_t(burnin+1:T)),'Smoothing',1600);
[~,C_hp]    = hpfilter(log(C_t(burnin+1:T)),'Smoothing',1600);
[~,I_hp]    = hpfilter(log(I_t(burnin+1:T)),'Smoothing',1600);
[~,L_hp]    = hpfilter(log(L_t(burnin+1:T)),'Smoothing',1600);
[~,Y2L_hp]  = hpfilter(log(Y_t(burnin+1:T)./L_t(burnin+1:T)),'Smoothing',1600);
[~,K_hp]    = hpfilter(log(K_t(burnin+1:T)),'Smoothing',1600);
[~,w_hp]    = hpfilter(log(w_t(burnin+1:T)),'Smoothing',1600);
[~,r_hp]    = hpfilter(log(r_t(burnin+1:T)),'Smoothing',1600);

% present business cycle statistics 
fprintf('\n');
fprintf('======================== \n');
fprintf('Final report\n');
fprintf('======================== \n');
fprintf('Convergence criterion: \n');
fprintf('Error: %.9f \n', err_ge);

fprintf('\n');
fprintf('======================== \n');
fprintf('Business cycle statistics for the raw time series\n');
fprintf('======================== \n');
fprintf('mean log(output): %.4f \n', mean(log(Y_t)));
fprintf('st. dev. log(output): %.4f \n', std(log(Y_t)));
fprintf('skewness log(output): %.4f \n', skewness(log(Y_t)));
fprintf('------------------------ \n');
fprintf('mean log(investment): %.4f \n', mean(log(I_t)));
fprintf('st. dev. log(investment): %.4f \n', std(log(I_t)));
fprintf('skewness log(investment): %.4f \n', skewness(log(I_t)));
fprintf('------------------------ \n');
fprintf('mean log(consumption): %.4f \n', mean(log(C_t)));
fprintf('st. dev. log(consumption): %.4f \n', std(log(C_t)));
fprintf('skewness log(consumption): %.4f \n', skewness(log(C_t)));

fprintf('\n');
fprintf('======================== \n');
fprintf('Business cycle statistics for the HP-filtered time series\n');
fprintf('======================== \n');
fprintf('mean log(output): %.4f \n', mean((Y_hp)));
fprintf('st. dev. log(output): %.4f \n', std((Y_hp)));
fprintf('skewness log(output): %.4f \n', skewness((Y_hp)));
fprintf('------------------------ \n');
fprintf('mean log(investment): %.4f \n', mean((I_hp)));
fprintf('st. dev. log(investment): %.4f \n', std((I_hp)));
fprintf('skewness log(investment): %.4f \n', skewness((I_hp)));
fprintf('------------------------ \n');
fprintf('mean log(consumption): %.4f \n', mean((C_hp)));
fprintf('st. dev. log(consumption): %.4f \n', std((C_hp)));
fprintf('skewness log(consumption): %.4f \n', skewness((C_hp)));

% prepare full set of statistics 
varNames = {'Output'; 'Consumption'; 'Investment'; 'Labour'; 'Y2L'; 'Capital'; 'w'; 'r'; 'A'};
relvolY = [std(Y_hp)/std(Y_hp); std(C_hp)/std(Y_hp); std(I_hp)/std(Y_hp); ...
    std(L_hp)/std(Y_hp); std(Y2L_hp)/std(Y_hp); std(K_hp)/std(Y_hp); ...
    std(w_hp)/std(Y_hp); std(r_hp)/std(Y_hp); std(vgrida(ishock_t(burnin+1:T)))/std(Y_hp)];
corrY = [corr(Y_hp,Y_hp); corr(Y_hp,C_hp); corr(Y_hp,I_hp); ...
    corr(Y_hp,L_hp); corr(Y_hp,Y2L_hp); corr(Y_hp,K_hp); ...
    corr(Y_hp,w_hp); corr(Y_hp,r_hp); corr(Y_hp,vgrida(ishock_t(burnin+1:T))')];
autocorr = [corr(Y_hp(1:end-1),Y_hp(2:end)); corr(C_hp(1:end-1),C_hp(2:end)); ...
    corr(I_hp(1:end-1),I_hp(2:end)); corr(L_hp(1:end-1),L_hp(2:end)); ... 
    corr(Y2L_hp(1:end-1),Y2L_hp(2:end)); corr(K_hp(1:end-1),K_hp(2:end)); ...
    corr(w_hp(1:end-1),w_hp(2:end)); corr(r_hp(1:end-1),r_hp(2:end)); ... 
    corr(vgridA(ishock_t(burnin+1:T-1))',vgridA(ishock_t(burnin+2:T))')];

% Create and display table for full set of statistics
fprintf('\n');
fprintf('======================== \n');
fprintf('Full Business Cycle Statistics on HP-filtered time series \n');
fprintf('======================== \n');
RBC_stats = table(relvolY, corrY, autocorr, 'RowNames', varNames, ...
    'VariableNames', {'Rel_Vol_Y', 'Corr_Y', 'Autocorr'});
disp(RBC_stats);
fprintf('Notes: Rel_Vol_Y = σ(x)/σ(y); Corr_Y = corr(x,y); Autocorr = corr(x_t, x_{t-1})\n');
%%
%=========================  
% dynamic consistency report
%========================= 
fprintf('\n');
fprintf('======================== \n');
fprintf('dynamic consistency report for rtm \n');
fprintf('======================== \n');
disp(['max absolute error (in pct. of steady-state K): ', num2str(100*max(abs(err_K))/ss.K),'%']);
disp(['root mean sqaured error (in pct. of steady-state K): ', num2str(100*sqrt(mean(err_K.^2))/ss.K),'%']);
fprintf('\n');
figure;
hist(100*err_K/ss.K,100);
grid on;
xlim([-1,1]);
xlabel("Dynamic consistency error (in pct. of steady-state K)")
ax = gca;
ax.FontSize = 12; 
set(gcf, 'Units', 'inches', 'Position', [1 1 5 4]);
exportgraphics(gcf, '../Figures/err_hist.pdf', 'ContentType', 'vector');
%%
%=========================
% fitting capital LoM into the linear specification
%=========================
% subset capital path 
endostate = K_t(burnin+1:end-2);
exostate = vgridA(ishock_t(burnin+1:end-1))';
endostateprime = K_t(burnin+2:end-1);

% state contingent regressions
idx_low = (exostate == vgridA(1));  % A = 0.99
idx_high = (exostate == vgridA(2)); % A = 1.01
x_low = [ones(sum(idx_low),1), log(endostate(idx_low))]; % independent variables
y_low = log(endostateprime(idx_low)); % dependent variable
[coeff_low,bint_low,r_low,rint_low,R_low] = regress(y_low,x_low);
x_high = [ones(sum(idx_high),1), log(endostate(idx_high))]; % independent variables
y_high = log(endostateprime(idx_high)); % dependent variable
[coeff_high,bint_high,r_high,rint_high,R_high] = regress(y_high,x_high);

% report
fprintf('======================== \n');
fprintf('Fitting the true LoM into the log-linear specification\n');
fprintf('======================== \n');
fprintf('\n');
fprintf('======================== \n');
fprintf('Low Productivity State (A = %.2f)\n', vgridA(1));
fprintf('======================== \n');
disp(fitlm(x_low, y_low, 'Intercept', false));
disp(['R-sq: ', num2str(R_low(1))])
fprintf('======================== \n');
fprintf('High Productivity State (A = %.2f)\n', vgridA(2));
fprintf('======================== \n');
disp(fitlm(x_high, y_high, 'Intercept', false));
disp(['R-sq: ', num2str(R_high(1))])

% recover the implied linear dynamics
samplePeriod = 500:1000;
recovered = zeros(1, length(endostate)+1);
recovered(1) = endostate(1);  % initiate from actual K(1)
for t = 1:length(endostate)
    if exostate(t) == vgridA(1)
        recovered(t+1) = exp(coeff_low' * [1; log(recovered(t))]);
    else
        recovered(t+1) = exp(coeff_high' * [1; log(recovered(t))]);
    end
end
err_lom_K = endostate(samplePeriod) - recovered(samplePeriod)';

% plot
figure;
plot(samplePeriod,endostate(samplePeriod),'Color','red','LineWidth',1.5);
hold on;
plot(samplePeriod,recovered(samplePeriod),'Color','blue','LineStyle','--','LineWidth',1.5);
hold off;
grid on;
ylabel('K');
xlabel('Transition Path');
legend("True LoM","Linear LoM","location","southwest","FontSize",8);
ax = gca;
ax.FontSize = 15; 
set(gcf, 'Units', 'inches', 'Position', [1 1 5 4]);
exportgraphics(gcf, '../Figures/lom_K.pdf', 'ContentType', 'vector');

figure;
plot(samplePeriod,err_lom_K,'LineWidth',1.5);grid on;
ylabel('Error in Linear LoM (K)');
xlabel('Transition Path');
yline(0,'k--','LineWidth',1.5);
ax = gca;
ax.FontSize = 15; 
set(gcf, 'Units', 'inches', 'Position', [1 1 5 4]);
exportgraphics(gcf, '../Figures/err_lom_K.pdf', 'ContentType', 'vector');
%%
%=========================  
% fitting wage dynamics into the linear specification 
%=========================
% subset wage path 
endostate = K_t(burnin+1:end-2);
exostate = vgridA(ishock_t(burnin+1:end-1))';
endoprice = w_t(burnin+1:end-1);

% state contingent regressions
idx_low = (exostate == vgridA(1));  % A = 0.99
idx_high = (exostate == vgridA(2)); % A = 1.01
x_low = [ones(sum(idx_low),1), log(endostate(idx_low))]; % independent variables
y_low = log(endoprice(idx_low)); % dependent variable
[coeff_low,bint_low,r_low,rint_low,R_low] = regress(y_low,x_low);
x_high = [ones(sum(idx_high),1), log(endostate(idx_high))]; % independent variables
y_high = log(endoprice(idx_high)); % dependent variable
[coeff_high,bint_high,r_high,rint_high,R_high] = regress(y_high,x_high);

% report
fprintf('======================== \n');
fprintf('Fitting the true LoM into the log-linear specification\n');
fprintf('======================== \n');
fprintf('\n');
fprintf('======================== \n');
fprintf('Low Productivity State (A = %.2f)\n', vgridA(1));
fprintf('======================== \n');
disp(fitlm(x_low, y_low, 'Intercept', false));
disp(['R-sq: ', num2str(R_low(1))])
fprintf('======================== \n');
fprintf('High Productivity State (A = %.2f)\n', vgridA(2));
fprintf('======================== \n');
disp(fitlm(x_high, y_high, 'Intercept', false));
disp(['R-sq: ', num2str(R_high(1))])

% recover the implied linear dynamics
recovered = zeros(1, length(endostate));
for t = 1:length(endostate)
    if exostate(t) == vgridA(1)
        recovered(t) = exp(coeff_low' * [1; log(endostate(t))]);
    else
        recovered(t) = exp(coeff_high' * [1; log(endostate(t))]);
    end
end
err_lom_w = endoprice(samplePeriod) - recovered(samplePeriod)';

% plot
samplePeriod = 500:1000;
figure;
plot(samplePeriod,endoprice(samplePeriod),'Color','red','LineWidth',1.5);
hold on;
plot(samplePeriod,recovered(samplePeriod),'Color','blue','LineStyle','--','LineWidth',1.5);
hold off;
grid on;
ylabel('w');
xlabel('Transition Path');
legend("True LoM","Linear LoM","location","northeast","FontSize",8);
ax = gca;
ax.FontSize = 15; 
set(gcf, 'Units', 'inches', 'Position', [1 1 5 4]);
exportgraphics(gcf, '../Figures/lom_w.pdf', 'ContentType', 'vector');

figure;
plot(samplePeriod,err_lom_w,'LineWidth',1.5);grid on;
ylabel('Error in Linear LoM (w)');
xlabel('Transition Path');
yline(0,'k--','LineWidth',1.5);
ax = gca;
ax.FontSize = 15; 
set(gcf, 'Units', 'inches', 'Position', [1 1 5 4]);
exportgraphics(gcf, '../Figures/err_lom_w.pdf', 'ContentType', 'vector');

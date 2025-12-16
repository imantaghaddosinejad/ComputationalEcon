%% SOLVING KRUSELL AND SMITH (1998) WITH ENDOGENOUS LABOUR SUPPLY %%%%%%%%%
%
% 2025.16.12
% Author @ Iman Taghaddosinejad (https://github.com/imantaghaddosinejad)
%
% This script computes the Euler equation (EE) errors along the transition
% path for the converged GE price path.
%
% Reference:
% Lee, H. (2025). "Global Nonlinear Solutions in Sequence Space and the 
% Generalized Transition Function."
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% HOUSEKEEPING %%

close all;
clc;
clear variables;
addpath('../Functions')
addpath('../Solutions')
addpath('../Figures')

%% LOAD SOLUTION %% 

load '../Solutions/ks1998endolabfrisch_bc.mat';

%% COMPUTE EULER EQUATION ERROR ALONG TRANSITION PATH %%

%=========================
% backward solution step
%=========================
% solve backwards along the transition path for optimal policy rule
% (derived from EE) given ocnverged prices and compare with solution policy
% rule for consumption.
for t = T:-1:1

    % current aggregate state 
    iA = ishock_t(t);

    % current prices 
    r = r_t(t);
    w = w_t(t);

    % future aggregate state (for beliefs) 
    ifuture = itransp(t); % future time-index 
    ifutureshock = ishock_t(ifuture); % future shock 
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
                Klow(Klow>=length(index)) = length(index)-1; % snap to second to last period if location is above candidate set
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
    n = (w*mgridz)./(p.eta*c); % labour supply (using pre-computed matrix)
    mlambda_new_temp = 1./((1+r)*mgrida + w*mgridz.*n - squeeze(mpolaprime_t(:,:,t))) - mexp; % nnc as residual
    mpolaprime_new_temp = (1+r)*mgrida + w*mgridz.*n - c; % savings

    % frictions (enforce borrowing constraint)
    mlambda_new_temp(mpolaprime_new_temp>p.agridmin) = 0;
    c = (c - (p.agridmin-mpolaprime_new_temp)).*(mpolaprime_new_temp<=p.agridmin) ...
        + c.*(mpolaprime_new_temp>p.agridmin);
    mpolaprime_new_temp(mpolaprime_new_temp<=p.agridmin) = p.agridmin;

    % update 
    mlambda_new_t(:,:,t) = mlambda_new_temp;
    mpolaprime_new_t(:,:,t) = mpolaprime_new_temp;
    mpolc_new_t(:,:,t) = c; % consumption policy consistent with EE
    mpoln_t(:,:,t) = n; 
end

%=========================
% Euler equation error
%=========================
EE = (mpolc_new_t - mpolc_t)./mpolc_t;

%% 
%=========================
% visualise EE error for given state
%=========================
% the individual household state needs to be specified.
% the monotonicity result is robust over the choice of different individual states.
iz = floor(p.Nz/2);
ik = floor(p.Na/2);

ts1 = K_t(burnin+1:T);
ts2 = squeeze(EE(ik,iz,burnin+1:T)); 

% plot
figure;
scatter(1:length(ts2),log(abs(ts2)),18);
xlabel("Period");
ylabel("Euler equation error (in Log)");
hold off;
grid on;
xlim([1,length(ts2)]);
ax = gca;
ax.FontSize = 15; 
set(gcf, 'Units', 'inches', 'Position', [1 1 6 4]);
exportgraphics(gcf, '../Figures/ee_timeseries.pdf', 'ContentType', 'vector');

figure;
hist(log(abs(ts2)),100);
grid on;
xlabel("Euler equation error (in Log)");
ylabel("Distribution");
ax = gca;
ax.FontSize = 15; 
set(gcf, 'Units', 'inches', 'Position', [1 1 6 4]);
exportgraphics(gcf, '../Figures/ee_hist.pdf', 'ContentType', 'vector');

figure;
scatter(ts1,log(abs(ts2)),18);
xlabel("Aggregate capital stock");
ylabel("Euler equation error (in Log)");
grid on;
ax = gca;
ax.FontSize = 15; 
set(gcf, 'Units', 'inches', 'Position', [1 1 6 4]);
exportgraphics(gcf, '../Figures/ee.pdf', 'ContentType', 'vector');
%%
%=========================
% visualise EE error on average across all states
%=========================
% Average EE across all household states
ts1 = K_t(burnin+1:T);
ts2 = squeeze(mean(abs(EE(:,:,burnin+1:T)),[1,2])); % Average absolute error

% plot
figure;
scatter(1:length(ts2),log(abs(ts2)),18);
xlabel("Period");
ylabel("Mean Euler equation error (in Log)");
hold off;
grid on;
xlim([1,length(ts2)]);
ax = gca;
ax.FontSize = 15; 
set(gcf, 'Units', 'inches', 'Position', [1 1 6 4]);
exportgraphics(gcf, '../Figures/ee_avg_timeseries.pdf', 'ContentType', 'vector');

figure;
hist(log(abs(ts2)),100);
grid on;
xlabel("Mean Euler equation error (in Log)");
ylabel("Distribution");
ax = gca;
ax.FontSize = 15; 
set(gcf, 'Units', 'inches', 'Position', [1 1 6 4]);
exportgraphics(gcf, '../Figures/ee_avg_hist.pdf', 'ContentType', 'vector');

figure;
scatter(ts1,log(abs(ts2)),18);
xlabel("Aggregate capital stock");
ylabel("Mean Euler equation error (in Log)");
grid on;
ax = gca;
ax.FontSize = 15; 
set(gcf, 'Units', 'inches', 'Position', [1 1 6 4]);
exportgraphics(gcf, '../Figures/ee_avg.pdf', 'ContentType', 'vector');
%% SOLVING KRUSELL AND SMITH (1998) WITH ENDOGENOUS LABOUR SUPPLY %%%%%%%%%
%
% 2025.16.12
% Author @ Iman Taghaddosinejad (https://github.com/imantaghaddosinejad)
%
% This script checks the monotonicity of the policy function along with 
% the endogenous state in equilibrium.
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

% assign policy function(s)
iA = ishock_t;
vA  = vgridA(iA);
RHS = mpolc_t;

%%
%=========================
% check monotonicity condition
%=========================
% the individual household state needs to be specified
% the monotonicity result is robust over the choice of different individual states
iz = floor(p.Nz/2);
ik = floor(p.Na/2);

K_t_sample = K_t(burnin+1:T-burnin);
RHSsample = RHS(:,:,burnin+1:T-burnin);
tsimpathSample = ishock_t(burnin+1:T-burnin);

for iA = 1:p.NA
tempK = K_t_sample(tsimpathSample==iA);
tempRHS = squeeze(RHSsample(ik,iz,tsimpathSample==iA));
subplot(1,2,iA);
scatter(tempK,tempRHS,18);
xlabel("K");
ylabel("RHS of Euler");
grid on;
temptitle = append('A',num2str(iA));
title(temptitle);
ax = gca;
ax.FontSize = 15; 
end
set(gcf, 'Units', 'inches', 'Position', [1 1 10 6]);
exportgraphics(gcf, '../Figures/monotonicity.pdf', 'ContentType', 'vector');
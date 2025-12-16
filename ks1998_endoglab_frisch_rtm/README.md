Contains MATLAB code to globally solve Krusell and Smith (1998) with endogenous labour supply. The model is solved using the Repeated Transition Method (RTM) developed by Lee (2025). My extension is computational: I parallelize the backward solution step along the transition path to improve runtime performance.

### Directory Guide
`./Main` - Main MATLAB scripts to solve model and run diagnostics.  
  - `ks1998endoglabfrisch_ss.m` - Script to solve for steady-state.  
  - `ks1998endoglabfrisch_bc.m` - Script to globally solve dynamic model using parallelization.  
  - `ks1998endoglabfrisch_sequential_bc.m` - Script to globally solve dynamic model using sequential approach.  
  - `ks1998endoglabfrisch_ee.m` - Compute Euler equation errors.  
  - `ks1998endoglabfrisch_monotonicity.m` - Check monotonicity of value/policy functions in equilibrium.  
`./Functions` - Auxiliary functions used across main scripts.  
`./Solutions` - All relevant solutions as MATLAB data files.  
`./Figures` - Relevant figures for solutions.  

### References
Krusell, P., & Smith, A. A. Jr. (1998). Income and Wealth Heterogeneity in the Macroeconomy. Journal of Political Economy, 106(5), 867-896.  
Lee, H. (2025). Global Nonlinear Solutions in Sequence Space and the Generalized Transition Function. Working Paper.
%% Stationary Distribution of First Order Markov Process %%%%%%%%%%%%%%%%%%
%   Derive the Invariant Distribution Corresponding to a Transition Matrix 
%   
%   derivation uses P' * x = x, implying x is the invariant distribution
%
%   Args:
%       P: transition matrix
%       eigvlmethod: boolean to indicate using eigenvalue method (default: True)
%
%
function vPi = fnStationaryDist(P, eigvlmethod)
    if nargin < 2
        eigvlmethod = true;
    end
    if eigvlmethod
        [v, D] = eig(P');
        [~, idx] = max(abs(diag(D)));
        vPi = v(:, idx);
        vPi = vPi / sum(vPi);
    else
        x0 = zeros(length(P), 1);
        x0(1) = 1;
        err = 10;
        while err >= 1e-12
            x1 = P' * x0;            
            err = max(abs(x1 - x0)); 
            x0 = x1;
        end
        vPi = x0;
    end
end
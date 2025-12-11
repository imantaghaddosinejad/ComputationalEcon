%% Create Joint Transition Matrix %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% Compute Joint Transition Matrix Over (k,z) States 
%
% Args:
%   pNk: (acalar) number of states for k (capital)
%   pNz: (scalar) number of states for z (productivity)
%   mPolkprime: (matrix) (interpolated) policy rule for optimal investment 
%   vGridk: (vector) capital grid 
%   mPz: (matrix) probability transition matrix for productivity
%
% Returns:
%   mJointTrans: (matrix) joint transition matrix over (k,z)
%
function mJointTrans = fnTransMat(pNk, pNz, mPolkprime, vGridk, mPz)
    mJointTrans = zeros(pNk*pNz, pNk*pNz);
    for ik = 1:pNk
        for iz = 1:pNz
            iCurrState = (iz - 1)*pNk + ik; % corresponding i-th row for state (ik, iz)
            kprime = mPolkprime(ik, iz); % optimal savings given current state (ik, iz)
            [LB, UB, wtLB, wtUB] = fnInterp1dGrid(kprime, vGridk, pNk); % interpolate kprime on capital grid
            
            for izfuture = 1:pNz
                iFutureStateLB = (izfuture - 1)*pNk + LB; % index for LB - future state 
                iFutureStateUB = (izfuture - 1)*pNk + UB; % index for UB - future state 
                mJointTrans(iCurrState, iFutureStateLB) = 1 * mPz(iz, izfuture) * wtLB; % prob. transition to LB
                mJointTrans(iCurrState, iFutureStateUB) = 1 * mPz(iz, izfuture) * wtUB; % prob. transition to UB
            end
        end
    end
end
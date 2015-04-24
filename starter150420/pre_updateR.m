function [R_W1] = pre_updateR( data , R_W1, signal, omega, outputSize) 
%PRE_UPDATER Summary of this function goes here
%   Detailed explanation goes here
    reward = zeros (outputSize, 1);
    mat = data' * R_W1;
    output = mat
    [MAX,MaxIndex] = max(output);
    if (signal == MaxIndex)
        reward(MaxIndex,1) = 1;
    else
        reward(MaxIndex,1) = -1;
    end
    delta = omega * data * reward';
    R_W1 = R_W1 + delta;
    
    %=============Debug========
%     output
%     MaxIndex
%     signal
      reward
%     data
%     delta
    %===================

end

function [R_W1] = updateR( data , R_W1, signal, omega, outputSize) 
%TRAINR Summary of this function goes here
%   Detailed explanation goes here
    reward = zeros (outputSize, 1);
    mat = data' * R_W1
    output = mat;
    [MAX,MaxIndex] = max(output);
    if (signal == MaxIndex)
        reward(MaxIndex,1) = 1;
    else 
        reward(MaxIndex,1) = -1;
    end
    reward(5,1)=0; % ignore the reward for white backgroud
    delta = omega * data * reward';
    R_W1 = R_W1 + delta;
    
    %=============Debug========
%     output
%     MaxIndex
%     signal
     if MaxIndex ~= 5
         reward
         signal
     end
%     data
%     delta
    %===================

end
%initialize
[inputSize,testSize] = size(Feature_DTest_white); % testSize = patchNum
outputSize = 5; %4 directions and a backgroud color 
r2  = sqrt(6) / sqrt(inputSize+outputSize+1);   % we'll choose weights uniformly from the interval [-r, r]
R_W1 = rand(inputSize, outputSize ) * 2 * r2 - r2;
%b1 = zeros(outputSize, 1);
omega = 1;
trainTimes = 100;

for k = 1: trainTimes
    data = Feature_DTest_white;
    for j =1 : testSize
        % a signal represents a type of a patch (An original image implies a direction type). 
        % label and signal are not the same.

        %In this case, pre-training only use white patch.
        
         signal = 5;
        %==========for debug=========
%         signal 
%         j
%         data(:,j)
        %=============
        R_W1 = pre_updateR( data(:,j) , R_W1, signal, omega, outputSize);
    end
end



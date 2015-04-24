%initialize
[inputSize,testSize] = size(Feature_DTest); % testSize = patchNum
outputSize = 5; % 4 directions and a background colour

%R_W1 is pretrained 
%r2  = sqrt(6) / sqrt(inputSize+outputSize+1);   % we'll choose weights uniformly from the interval [-r, r]
%R_W1 = rand(inputSize, outputSize ) * 2 * r2 - r2;

omega = 0.001;
trainTimes = 10;

for k = 1: trainTimes
    data = Feature_DTest;
    for j =1 : testSize
        % signal is the type of each patches (An original image implies a direction type). 
        % label and signal are not the same.
        testlabel = label_DTest(j);
        
        signal = 0; 
        if (mod(testlabel,4) ==0) % wait
            signal = 4;
        else if (mod(testlabel,4) ==1) % right
            signal = 3;
            else if (mod(testlabel,4) ==2) % left
                signal = 1; 
                else if (mod(testlabel,4) == 3) % straight
                    signal = 2;
                    % white  images are pre-trained.
                end
            end
        end
        end
        %==========for debug=========
%          signal 
%          j
%         data(:,j)
        %=============
        R_W1 = updateR( data(:,j) , R_W1, signal, omega, outputSize);
    end
end




function [Feature_RTest,label_RTest] = testmean(im,W1,hiddenSize,visibleSize,opttheta)
%TESTMEAN Summary of this function goes here
%   Get the mean value of features for each individual test images]
    testNum = size (im,3);
    patchNum = 1200;
    Feature_RTest = zeros (hiddenSize,testNum);
    
    [patches,label] = testIMAGES(im,patchNum);

    mat  = W1 * patches;
    
    b1 = reshape ( opttheta(2*hiddenSize*visibleSize+1:2*hiddenSize*visibleSize+hiddenSize), hiddenSize, 1 );       
    f = zeros(size(mat));
    
    for j = 1 : hiddenSize 
        f(j,:) = sigmoid(mat(j,:) + b1(j,1));
    end
    

    % calculate the mean value.
    cnt = zeros (testNum);
    for patchIndex = 1 : patchNum
        labelIndex = label(patchIndex); % labelindex is the index of the images.
        cnt(labelIndex)= cnt(labelIndex) + 1;        
        Feature_RTest(:,labelIndex) =  Feature_RTest(:,labelIndex) + f(:,patchIndex);
    end

    for labelIndex = 1 : testNum
        Feature_RTest(:,labelIndex) = Feature_RTest(:,labelIndex) / cnt(labelIndex);
    end 
    label_RTest = label;
end

function sigm = sigmoid(x)
  
    sigm = 1 ./ (1 + exp(-x));
end

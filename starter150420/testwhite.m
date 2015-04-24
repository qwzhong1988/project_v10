function [Feature_DTest,fwos] = testwhite(im,W1,hiddenSize,visibleSize,opttheta)
%TESTWHITE Summary of this function goes here
%   Test all generated patches]
    testNum = size (im,3);
    %patchNum = 600;
    Feature_DTest = zeros (hiddenSize,testNum);
    
    patches = reshape(im,64,1); 

    mat  = W1 * patches;
    
    b1 = reshape ( opttheta(2*hiddenSize*visibleSize+1:2*hiddenSize*visibleSize+hiddenSize), hiddenSize, 1 );       
    f = zeros(size(mat));
    fwos=f;
    
    for j = 1 : hiddenSize 
        f(j,:) = sigmoid( mat(j,:) + b1(j,1));
        fwos(j,:)=mat(j,:) + b1(j,1);% for debug
    end

    Feature_DTest = f;
end

function sigm = sigmoid(x)
    k = 1;
    sigm = 1 ./ (1 + exp(-k*x));
end

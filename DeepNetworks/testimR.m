[inputSize,testSize] = size(Feature_RTest);
Action_RTest = zeros (outputSize, testSize);
for i = 1 : testSize
   	Action_RTest (:,i)= R_W1' * Feature_RTest(:,i);
end
[MAX,Index] = max(Action_RTest(1:outputSize-2,1:testSize))
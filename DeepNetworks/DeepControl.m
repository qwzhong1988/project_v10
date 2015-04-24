%% Deep Control 
%  modified CS294A/CS294W Stacked Autoencoder Exercise
%
%%======================================================================
%% STEP 0: Here we provide the relevant parameters values that will
%  allow your sparse autoencoder to get good filters; you do not need to 
%  change the parameters below.

inputSize = 8 * 8;
numClasses = 5;
hiddenSizeL1 = 25;    % Layer 1 Hidden Size
hiddenSizeL2 = 13;    % Layer 2 Hidden Size
outputSize = 5; % 4 directions and a background colour

sparsityParam = 0.01;   % desired average activation of the hidden units.
                       % (This was denoted by the Greek alphabet rho, which looks like a lower-case "p",
		               %  in the lecture notes). 
lambda = 0.0001;         % weight decay parameter
beta = 3;              % weight of sparsity penalty term   


%%======================================================================
%% STEP 1: Implement sampleIMAGES
%
%  After implementing sampleIMAGES, the display_network command should
%  display a random sample of 200 patches from the dataset

patches = sampleIMAGES;
trainData = patches;
display_network(patches(:,randi(size(patches,2),200,1)),8);

%%======================================================================
%% STEP 2: Train the first sparse autoencoder
%  This trains the first sparse autoencoder on the unlabelled STL training
%  images.
%  If you've correctly implemented sparseAutoencoderCost.m, you don't need
%  to change anything here.


%  Randomly initialize the parameters
sae1Theta = initializeParameters(hiddenSizeL1, inputSize);

%% ---------------------- YOUR CODE HERE  ---------------------------------
%  Instructions: Train the first layer sparse autoencoder, this layer has
%                an hidden size of "hiddenSizeL1"
%                You should store the optimal parameters in sae1OptTheta

%  Use minFunc to minimize the function
addpath minFunc/
options.Method = 'lbfgs'; % Here, we use L-BFGS to optimize our cost
% function. Generally, for minFunc to work, you
% need a function pointer with two outputs: the
% function value and the gradient. In our problem,
% sparseAutoencoderCost.m satisfies this.
options.maxIter = 400;	  % Maximum number of iterations of L-BFGS to run
% options.maxIter = 1;	  % Maximum number of iterations of L-BFGS to run

options.display = 'on';

[sae1OptTheta, cost] = minFunc( @(p) sparseAutoencoderCost(p, ...
                                                           inputSize, hiddenSizeL1, ...
                                                           lambda, sparsityParam, ...
                                                           beta, trainData), ...
                               sae1Theta, options);


% -------------------------------------------------------------------------



%%======================================================================
%% STEP 2: Train the second sparse autoencoder
%  This trains the second sparse autoencoder on the first autoencoder
%  featurse.
%  If you've correctly implemented sparseAutoencoderCost.m, you don't need
%  to change anything here.
 
load('IMAGES_DTest.mat')% 4 sketches

patchNum = 12000;
[trainDataL2,label] = testIMAGES(IMAGES_DTest,patchNum);  

[sae1Features] = feedForwardAutoencoder(sae1OptTheta, hiddenSizeL1, ...
                                        inputSize, trainDataL2);

%  Randomly initialize the parameters
sae2Theta = initializeParameters(hiddenSizeL2, hiddenSizeL1);

%% ---------------------- YOUR CODE HERE  ---------------------------------
%  Instructions: Train the second layer sparse autoencoder, this layer has
%                an hidden size of "hiddenSizeL2" and an inputsize of
%                "hiddenSizeL1"
%
%                You should store the optimal parameters in sae2OptTheta

[sae2OptTheta, cost] = minFunc( @(p) sparseAutoencoderCost(p, ...
                                                           hiddenSizeL1, hiddenSizeL2, ...
                                                           lambda, sparsityParam, ...
                                                           beta, sae1Features), ...
                               sae2Theta, options);

% -------------------------------------------------------------------------
[sae2Features] = feedForwardAutoencoder(sae2OptTheta, hiddenSizeL2, ...
                                        hiddenSizeL1, sae1Features);

%%======================================================================
%% STEP 5: Visualization

W1L1 = reshape(sae1OptTheta(1:hiddenSizeL1*inputSize), hiddenSizeL1, inputSize);
display_network(W1L1', 12);
                
                print -djpeg weights.jpg   % save the visualization to a file
                
W1L2 = reshape(sae2OptTheta(1:hiddenSizeL2*hiddenSizeL1), hiddenSizeL2, hiddenSizeL1);

W = W1L2*W1L1;
display_network(W', 12);
                                
                print -djpeg weights2.jpg   % save the visualization to a file
                

% -------------------------------------------------------------------------                
%%======================================================================
% %% STEP 6: Train the Reinforcement Learning
% % Feature_DTest = sae2Features;
% % label_DTest = label;
% % r2  = sqrt(6) / sqrt(hiddenSizeL2+outputSize+1);   % we'll choose weights uniformly from the interval [-r, r]
% % R_W1 = rand(hiddenSizeL2, outputSize ) * 2 * r2 - r2;
% % trainR;
% load('IMAGES_DTestL3'); %values should be 0-1 size:n*m*num.
% 
% IMAGES_DTestL3 = max(min(IMAGES_DTestL3,0.9),0.1);
% 
% trainDataL3 = reshape(IMAGES_DTestL3,size(IMAGES_DTestL3,1)*size(IMAGES_DTestL3,2),size(IMAGES_DTestL3,3));
% [sae1FeaturesL3] = feedForwardAutoencoder(sae1OptTheta, hiddenSizeL1, ...
%                                         inputSize, trainDataL3);
% [sae2FeaturesL3] = feedForwardAutoencoder(sae2OptTheta, hiddenSizeL2, ...
%                                         hiddenSizeL1, sae1FeaturesL3);
% 
% Feature_DTest = sae2FeaturesL3;
% label_DTest = [1,2,3,4,5];
% r2  = sqrt(6) / sqrt(hiddenSizeL2+outputSize+1);   % we'll choose weights uniformly from the interval [-r, r]
% R_W1 = rand(hiddenSizeL2, outputSize ) * 2 * r2 - r2;    
% trainR;

%----------
%%=======
%% Test 
% Test for layer one 
load('IMAGES_DTestL3'); %values should be 0-1 size:n*m*num.

IMAGES_DTestL3 = max(min(IMAGES_DTestL3,0.9),0.1);

trainDataL3 = reshape(IMAGES_DTestL3,size(IMAGES_DTestL3,1)*size(IMAGES_DTestL3,2),size(IMAGES_DTestL3,3));
for i= 1: 1000 
[sae1FeaturesL3] = feedForwardAutoencoder(sae1OptTheta, hiddenSizeL1, ...
                                        inputSize, trainDataL3(:,3));
 test_sae1FeaturesL3 (:,i)=       sae1FeaturesL3;                            
end                                    
% [sae2FeaturesL3] = feedForwardAutoencoder(sae2OptTheta, hiddenSizeL2, ...
%                                         hiddenSizeL1, sae1FeaturesL3);
%                                     
%   ts=1;
% Action_RTest = zeros (outputSize, ts);
% for j = 1 : ts
%         Action_RTest (:,j)  = R_W1' * Feature_DTest(:,j);
% end
% [MAX,Index] = max(Action_RTest(1:outputSize,1:5));

% -------------------------------------------------------------------------                
%%======================================================================
%% STEP 7: Test the Network
% % Test the network with 12 sketches
% load('IMAGES_RTest'); 
% % test skecthes one by one
% skpatchNum=1000;
% outputResults = zeros(skpatchNum,size(IMAGES_RTest,3));
% for i = 1 : size(IMAGES_RTest,3)
%     im = IMAGES_RTest(:,:,i) ;
%     [patches,~] = testIMAGES(im,skpatchNum); 
%     [Feature_RTest0]= feedForwardAutoencoder(sae1OptTheta, hiddenSizeL1, ...
%                                              inputSize, patches);
%     [Feature_RTest] = feedForwardAutoencoder(sae2OptTheta, hiddenSizeL2, ...
%                                         hiddenSizeL1, Feature_RTest0);
%                                     
%     [~,testSize] = size(Feature_RTest);
%     Action_RTest = zeros (outputSize, testSize);
%     for j = 1 : testSize
%         Action_RTest (:,j)  = R_W1' * Feature_RTest(:,j);
%     end
%     [MAX,Index] = max(Action_RTest(1:outputSize,1:testSize)); 
%     outputResults(:,i) = Index'; 
% end
% testim2;
% testimR;



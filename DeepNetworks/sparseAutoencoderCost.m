function [cost,grad] = sparseAutoencoderCost(theta, visibleSize, hiddenSize, ...
                                             lambda, sparsityParam, beta, data)

% visibleSize: the number of input units (probably 64) 
% hiddenSize: the number of hidden units (probably 25) 
% lambda: weight decay parameter
% sparsityParam: The desired average activation for the hidden units (denoted in the lecture
%                           notes by the greek alphabet rho, which looks like a lower-case "p").
% beta: weight of sparsity penalty term
% data: Our 64x10000 matrix containing the training data.  So, data(:,i) is the i-th training example. 
  
% The input theta is a vector (because minFunc expects the parameters to be a vector). 
% We first convert theta to the (W1, W2, b1, b2) matrix/vector format, so that this 
% follows the notation convention of the lecture notes. 

W1 = reshape(theta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
W2 = reshape(theta(hiddenSize*visibleSize+1:2*hiddenSize*visibleSize), visibleSize, hiddenSize);
b1 = theta(2*hiddenSize*visibleSize+1:2*hiddenSize*visibleSize+hiddenSize);
b2 = theta(2*hiddenSize*visibleSize+hiddenSize+1:end);

% Cost and gradient variables (your code needs to compute these values). 
% Here, we initialize them to zeros. 
cost = 0;
W1grad = zeros(size(W1)); 
W2grad = zeros(size(W2));
b1grad = zeros(size(b1)); 
b2grad = zeros(size(b2));

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute the cost/optimization objective J_sparse(W,b) for the Sparse Autoencoder,
%                and the corresponding gradients W1grad, W2grad, b1grad, b2grad.
%
% W1grad, W2grad, b1grad and b2grad should be computed using backpropagation.
% Note that W1grad has the same dimensions as W1, b1grad has the same dimensions
% as b1, etc.  Your code should set W1grad to be the partial derivative of J_sparse(W,b) with
% respect to W1.  I.e., W1grad(i,j) should be the partial derivative of J_sparse(W,b) 
% with respect to the input parameter W1(i,j).  Thus, W1grad should be equal to the term 
% [(1/m) \Delta W^{(1)} + \lambda W^{(1)}] in the last block of pseudo-code in Section 2.2 
% of the lecture notes (and similarly for W2grad, b1grad, b2grad).
% 
% Stated differently, if we were using batch gradient descent to optimize the parameters,
% the gradient descent update to W1 would be W1 := W1 - alpha * W1grad, and similarly for W2, b1, b2. 
% 

%J_sparse(W,b) = J(W,b) + beta * sum(KL(p||cap(p)_j))    j=1:s2 
%              = J(W,b)+ sparsitycost
%              = ouputcost + regularizationcost + sparsitycost

% a denote the activation (meaning output value)
% z denote the total wighted sum of inputs
% f(z) is the sigmoid function

% z2 = W1*x + b1;
% a2 = f(z2);
% z3 = W2*a2 + b2;
% hW,b(x) = a3 = f(z3)

numSamples = size(data,2);

z2 = W1 * data + repmat(reshape(b1, hiddenSize, 1), 1, numSamples); % hiddenSize * numSamples
a2 = sigmoid(z2); 
z3 = W2 * a2 + repmat(reshape(b2, visibleSize, 1), 1, numSamples); % visibleSize * numSamples
a3 = sigmoid(z3);

%calculate J(W,b)
%halferrorcost J(W,b;x,y)= 1/2 ||hW,b(x)-y||^2
%m examples 
%outputcost(average sum-of-squares error term) = 1/m * sum(J(W,b;x,y));
%regularizarionCost = lambda/2 * sumlayer(sumi(sumj(w)^2))

halferrorcost = ((data - a3).^2 )./2;
outputCost = sum(sum(halferrorcost))/ numSamples;
regularizationCost = lambda / 2 * (sum(sum((W1.^2))) + sum(sum(W2.^2)));

%calculate sparsityCost
%cap(p)_j = 1/m * sum (a2(x))
%KL(p||cap(p)_j)= p*log (p/cap(p)_j) + (1-p)*log((1-p)/(1-cap(p)_j)

caprho = sum(a2, 2) / numSamples;
sparsityCost = beta * sum(sparsityParam * log(sparsityParam ./ caprho) + (1-sparsityParam) * log((1-sparsityParam) ./ (1-caprho)));

cost = outputCost + regularizationCost + sparsityCost;

%calculating the grad of weights.
delta3 = - (data - a3) .* a3 .* (1 - a3); % visibleSize * numSamples

W2grad = delta3 * a2' / numSamples + lambda * W2; % visibleSize * hiddenSize
b2grad = sum(delta3, 2) / numSamples;

dSparsity = repmat(reshape(beta * (-sparsityParam./caprho + (1-sparsityParam)./(1-caprho)), hiddenSize, 1), 1, numSamples); % hiddenSize * numSamples
delta2 = (W2' * delta3 + dSparsity) .* a2 .* (1 - a2); % hiddenSize * numSamples

W1grad = delta2 * data' / numSamples + lambda * W1; % hiddenSize * visibleSize
b1grad = sum(delta2, 2) / numSamples;


%-------------------------------------------------------------------
% After computing the cost and gradient, we will convert the gradients back
% to a vector format (suitable for minFunc).  Specifically, we will unroll
% your gradient matrices into a vector.

grad = [W1grad(:) ; W2grad(:) ; b1grad(:) ; b2grad(:)];

end

%-------------------------------------------------------------------
% Here's an implementation of the sigmoid function, which you may find useful
% in your computation of the costs and the gradients.  This inputs a (row or
% column) vector (say (z1, z2, z3)) and returns (f(z1), f(z2), f(z3)). 

function sigm = sigmoid(x)
    k = 1;
    sigm = 1 ./ (1 + exp(-k*x));
end


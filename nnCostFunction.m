function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
 
 #first stage is to use forward propagation
  X = [ones(m,1), X];  % Adding 1 as first column in X
  
  a1 = X; % 5000 x 401
  
  z2 = a1 * Theta1';  % m x hidden_layer_size == 5000 x 25
  a2 = sigmoid(z2); % m x hidden_layer_size == 5000 x 25
  a2 = [ones(size(a2,1),1), a2]; 
  # add a vector of ones as a bias unit to the hidden layer which results in a 5000x26 matrix
  z3 = a2 * Theta2';  % m x num_labels == 5000 x 10
  a3 = sigmoid(z3); #10 rows are outputted corresponding to the 10 different numbers
  
  h_x = a3; % m x num_labels == 5000 x 10
  #if there are initially 5000 exapmles then it makes sense for there to be 5000 answers
  %Converting y into vector of 0's and 1's for multi-class classification
 
  #create a matrix to sub into the cost function which contains the correct values for the train set
  y_train = (1:num_labels)==y; #5000 x 10
  
  %Costfunction Without regularization
  J = (1/m) * sum(sum((-y_train.*log(h_x))-((1-y_train).*log(1-h_x))));  %scalar
 
#####################################################
 #Gradient computation: Backpropagation algorithm 
  
  D3 = a3 - y_train; % 5000 x 10
  D2 = (D3 * Theta2) .* [ones(size(z2,1),1) sigmoidGradient(z2)]; % 5000 x 26
  D2 = D2(:,2:end); % 5000 x 25 %Removing delta2 for bias node
  
  Theta1_grad = (1/m) * (D2' * a1); % 25 x 401
  Theta2_grad = (1/m) * (D3' * a2); % 10 x 26
  
  
  %%%%%%%%%%%% Part 3: Adding Regularisation term in J and Theta_grad %%%%%%%%%%%%%
  J = J +(lambda/(2*m)) * (sum(sum(Theta1(:,2:end).^2)) + sum(sum(Theta2(:,2:end).^2))); %scalar
  
  Theta1_grad_reg_term = (lambda/m) * [zeros(size(Theta1, 1), 1) Theta1(:,2:end)]; % 25 x 401
  Theta2_grad_reg_term = (lambda/m) * [zeros(size(Theta2, 1), 1) Theta2(:,2:end)]; % 10 x 26
 
  Theta1_grad = Theta1_grad + Theta1_grad_reg_term;
  Theta2_grad = Theta2_grad + Theta2_grad_reg_term;

#gradients and cost function are then input to fmin to optimise theta values which will minise cost
% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end

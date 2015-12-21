function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%



J = sum((X*theta - y).^2 )/(2*m); % 1 by 1

% regulaization_penaty
theta_square = theta.^2;
regularization_penaty = 0;
for i=2:size(theta,1)
   regularization_penaty += theta_square(i);
endfor
regularization_penaty = regularization_penaty*lambda/ (2*m);
J += regularization_penaty;


grad = (X'*(X*theta - y))/m + theta*lambda/m; % columX(thetaRow) by 1
grad(1) = grad(1) - lambda*theta(1)/m; % 










% =========================================================================

grad = grad(:);

end

function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

            
% You need to return the following values correctly
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost function and gradient for collaborative
%               filtering. Concretely, you should first implement the cost
%               function (without regularization) and make sure it is
%               matches our costs. After that, you should implement the 
%               gradient and use the checkCostFunction routine to check
%               that the gradient is correct. Finally, you should implement
%               regularization.
%
% Notes: X - num_movies  x num_features matrix of movie features
%        Theta - num_users  x num_features matrix of user features
%        Y - num_movies x num_users matrix of user ratings of movies
%        R - num_movies x num_users matrix, where R(i, j) = 1 if the 
%            i-th movie was rated by the j-th user
%
% You should set the following variables correctly:
%
%        X_grad - num_movies x num_features matrix, containing the 
%                 partial derivatives w.r.t. to each element of X
%        Theta_grad - num_users x num_features matrix, containing the 
%                     partial derivatives w.r.t. to each element of Theta
%



J = sum(sum((((X*(Theta)' - Y).^2).*R)))/2 + lambda*sum(sum(Theta.^2))/2 + lambda*sum(sum(X.^2))/2;

for i = 1:size(X(:,1))
   X_grad(i,:) = (((Theta*X(i,:)')-Y(i,:)').*R(i,:)')'*Theta + lambda*X(i,:);
   % Theta*X(i,:) ---> all user's predict rate on movie i    
   % Y(i,:) ->  all user's actuall rate on movie i.
   % R(i,:) -> true/false for all user's rating on movie i
endfor

 
for j = 1:size(Theta(:,1))
   Theta_grad(j,:) = (((X*Theta(j,:)')-Y(:,j)).*R(:,j))'*X + lambda*Theta(j,:);
    % X*Theta(j,:) ---> all movie's predict rate by user j 
   % Y(:,j) ->  All movie's actual rate by user j .
   % R(:,j) -> true/false for all movie rating by user j
endfor



% =============================================================

grad = [X_grad(:); Theta_grad(:)];

end

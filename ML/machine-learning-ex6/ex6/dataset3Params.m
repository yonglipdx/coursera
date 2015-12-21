function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

first_loop = 1;
min_c = 0;
min_sigma = 0;
min_error = 0;
for c = [0.01 0.03 0.1 0.3 1 3 10 30]
   for sigma =[0.01 0.03 0.1 0.3 1 3 10 30]
      model= svmTrain(X, y, c, @(x1, x2) gaussianKernel(x1, x2, sigma)); 
      pred = svmPredict(model, Xval);
      current_error = mean(double(pred ~= yval));
      if (first_loop == 1)
         first_loop = 0;
         min_c = c;
         min_sigma = sigma;
         min_error = current_error
      elseif (current_error < min_error)
         min_c = c;
         min_sigma = sigma;
         min_error = current_error
      endif
   endfor
endfor

C = min_c
sigma = min_sigma






% =========================================================================

end

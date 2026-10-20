function [ordered_by_pls1, model, rmse_te, mae_te] = ...
    kernel_R2_cal(X, Y, Xtest, Ytest, Kfold, maxVars)

% Kernel ridge regression with RBF kernel.
% Wavelengths selected using PLS1 forward selection.
% Kernel hyperparameters selected by cross-validation.
%
% Maximum kernel matrix size: 500 x 500.

X = double(X);
Y = double(Y(:));
Xtest = double(Xtest);
Ytest = double(Ytest(:));

% ------------------------------------------------------------
% 1. Select wavelengths using the existing PLS1 method
% ------------------------------------------------------------

[ordered_by_pls1, pls1_keep] = ...
    r2_pls1_forward_selection( ...
    X, Y, 1:size(X,2), 1e-3, maxVars);

X = X(:, pls1_keep);
Xtest = Xtest(:, pls1_keep);

% ------------------------------------------------------------
% 2. Limit kernel training data to 500 observations
% ------------------------------------------------------------

maxKernelSize = 500;

if size(X,1) > maxKernelSize

    rng(42); % Reproducibility

    idx = randperm(size(X,1), maxKernelSize);

    X = X(idx,:);
    Y = Y(idx);

end

% ------------------------------------------------------------
% 3. Cross-validation for kernel hyperparameters
% ------------------------------------------------------------

% Candidate kernel widths
sigmaValues = logspace(-2, 2, 12);

% Candidate regularization parameters
lambdaValues = logspace(-5, 2, 10);

cv = cvpartition(size(X,1), 'KFold', Kfold);

bestRMSE = inf;
bestSigma = NaN;
bestLambda = NaN;

for s = 1:numel(sigmaValues)

    sigma = sigmaValues(s);

    for l = 1:numel(lambdaValues)

        lambda = lambdaValues(l);

        squaredErrors = [];

        for f = 1:Kfold

            trainIdx = training(cv, f);
            valIdx = test(cv, f);

            Xtr = X(trainIdx,:);
            Ytr = Y(trainIdx);

            Xval = X(valIdx,:);
            Yval = Y(valIdx);

            % Standardization within each fold
            muX = mean(Xtr,1);
            sdX = std(Xtr,0,1);
            sdX(sdX == 0) = 1;

            Xtr = (Xtr - muX)./sdX;
            Xval = (Xval - muX)./sdX;

            muY = mean(Ytr);
            YtrCentered = Ytr - muY;

            % Training kernel
            Dtr = pdist2(Xtr, Xtr, 'squaredeuclidean');

            Ktr = exp(-Dtr/(2*sigma^2));

            % Kernel ridge regression
            alpha = (Ktr + lambda*eye(size(Ktr,1))) ...
                    \ YtrCentered;

            % Validation kernel
            Dval = pdist2(Xval, Xtr, 'squaredeuclidean');

            Kval = exp(-Dval/(2*sigma^2));

            Ypred = Kval*alpha + muY;

            squaredErrors = [squaredErrors; ...
                             (Yval-Ypred).^2];

        end

        rmseCV = sqrt(mean(squaredErrors));

        if rmseCV < bestRMSE

            bestRMSE = rmseCV;
            bestSigma = sigma;
            bestLambda = lambda;

        end
    end
end

% ------------------------------------------------------------
% 4. Fit final kernel model
% ------------------------------------------------------------

muX = mean(X,1);
sdX = std(X,0,1);
sdX(sdX == 0) = 1;

Xscaled = (X-muX)./sdX;
XtestScaled = (Xtest-muX)./sdX;

muY = mean(Y);
Ycentered = Y-muY;

D = pdist2(Xscaled, Xscaled, 'squaredeuclidean');

K = exp(-D/(2*bestSigma^2));

alpha = (K + bestLambda*eye(size(K,1))) \ Ycentered;

% ------------------------------------------------------------
% 5. Independent test predictions
% ------------------------------------------------------------

Dtest = pdist2(XtestScaled, Xscaled, 'squaredeuclidean');

Ktest = exp(-Dtest/(2*bestSigma^2));

Ypred = Ktest*alpha + muY;

rmse_te = sqrt(mean((Ytest-Ypred).^2));
mae_te = mean(abs(Ytest-Ypred));

% ------------------------------------------------------------
% 6. Store model for predicting missing values
% ------------------------------------------------------------

model.Xtrain = Xscaled;
model.alpha = alpha;
model.sigma = bestSigma;
model.lambda = bestLambda;

model.muX = muX;
model.sdX = sdX;
model.muY = muY;

model.selectedWavelengths = pls1_keep;
model.rmseCV = bestRMSE;

end
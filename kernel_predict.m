function Ypred = kernel_predict(model, Xnew)

Xnew = double(Xnew);

% Use selected wavelengths
Xnew = Xnew(:,model.selectedWavelengths);

% Standardize using training parameters
Xnew = (Xnew-model.muX)./model.sdX;

% Kernel between new observations and training observations
D = pdist2(Xnew,model.Xtrain,'squaredeuclidean');

K = exp(-D/(2*model.sigma^2));

% Predictions
Ypred = K*model.alpha + model.muY;

end
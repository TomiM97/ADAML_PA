function K = kernelRBFTest(model)

ci           = model.center;
model.center = 0;
KCal         = kernelRBF(model);
model.center = ci;

if isdlarray(model.X)
    X   = extractdata(model.X);
    Xt  = extractdata(model.Xtest);
else
    X   = model.X;
    Xt  = model.Xtest;
end

ND   = normDiff(Xt, X);
if  strcmp(model.kernelType, "matern1/2") || strcmp(model.kernelType, "matern3/2") || strcmp(model.kernelType, "matern5/2") || model.family == 1
    PD   = pdist2(Xt, X, 'cityblock');
end

if model.family == 1

    % Gaussian
    K1   = exp(-ND/(2*(model.params(1)^2)));

    % Matern 1/2
    K2   = exp(-PD/model.params(2));

    % Matern 2/3
    K3   = (1 + ((sqrt(3) * PD)/model.params(3))) .* exp(- (sqrt(3) * PD)/model.params(3));

    % Matern 5/2
    K4   = (1 + ((sqrt(5) * PD)/model.params(4)) + ((5 * ND)/(3*model.params(4)^2))) .* exp(- (sqrt(5) * PD)/model.params(4));

    % Cauchy
    K5   = 1 ./ ( 1 + ( ND ./ model.params(5) ^ 2));

    % Family
    %K = model.params(6) .* K1 + model.params(7) .* K2 + model.params(8) .* K3 + ...
    %    model.params(9) .* K4 + model.params(10) .* K5;

    K =  K1 + K2 + K3 + K4 + K5;

elseif model.family == -1 

    if isdlarray(model.X)
        X   = extractdata(model.X);
        Xt  = extractdata(model.Xtest);
    else
        X   = model.X;
        Xt  = model.Xtest;
    end

    [noRows, noVars] = size(X);
    [noRowsTest, noVars] = size(Xt);
    K = zeros(noRowsTest, noRows);

    if strcmp(model.kernelType, "gaussian")
        for i = 1:noVars
            K = K + exp( -normDiff(Xt(:,i), X(:,i))/(2*model.params(i)^2));
        end
    elseif strcmp(model.kernelType, "matern1/2")
        for i = 1:noVars
            K   = K + exp(-pdist2(Xt(:,i), X(:,i), 'cityblock')/model.params(i));
        end
    elseif strcmp(model.kernelType, "matern3/2")
        for i = 1:noVars
            PD  = pdist2(Xt(:,i), X(:,i), 'cityblock');
            K   = K + (1 + ((sqrt(3) * PD)/model.params(i))) .* exp(- (sqrt(3) * PD)/model.params(i));
        end
    elseif strcmp(model.kernelType, "matern5/2")
        for i = 1:noVars
            PD  = pdist2(Xt(:,i), X(:,i), 'cityblock');
            ND  = normDiff(Xt(:,i), X(:,i));
            K   = K + (1 + ((sqrt(5) * PD)/model.params(i)) + ((5 * ND)/(3*model.params(i)^2))) .* exp(- (sqrt(5) * PD)/model.params(i));
        end
    elseif strcmp(model.kernelType, "cauchy")
        for i = 1:noVars
            ND  = normDiff(Xt(:,i), X(:,i));
            K   = K + 1 ./ ( 1 + ( ND ./ model.params(i) ^ 2));
        end
    end

else

    if strcmp(model.kernelType, "gaussian")
       K = exp(-ND/(2*(model.params(1)^2)));

    elseif strcmp(model.kernelType, "matern1/2")
       K   = exp(-PD/model.params(1));

    elseif strcmp(model.kernelType, "matern3/2")
       K   = (1 + ((sqrt(3) * PD)/model.params(1))) .* exp(- (sqrt(3) * PD)/model.params(1)); 
       
    elseif strcmp(model.kernelType, "matern5/2")
       K   = (1 + ((sqrt(5) * PD)/model.params(1)) + ((5 * ND)/(3*model.params(1)^2))) .* exp(- (sqrt(5) * PD)/model.params(1));

    elseif strcmp(model.kernelType, "cauchy")
       K   = 1 ./ ( 1 + ( ND ./ model.params(1) ^ 2));
    end

end

if model.center == 1
    n        =  length(KCal(:,1));
    oneN     =  ones(n,n)/n;
    nTest    =  length(model.Xtest(:,1));
    oneNTest =  ones(nTest, n)/n;
    K        =  K - oneNTest*KCal - K*oneN + oneNTest*KCal*oneN;
end

end
function K = kernelRBF(model)

if model.family == 1

    if isdlarray(model.X)
        X = extractdata(model.X);
    else
        X = model.X;
    end

    ND = normDiff(X, X);
    PD = pdist2(X, X, 'cityblock');

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

    K = K1 + K2 + K3 + K4 + K5;

elseif model.family == -1

    if isdlarray(model.X)
        X = extractdata(model.X);
    else
        X = model.X;
    end

    [noRows, noVars] = size(X);
    K = zeros(noRows, noRows);

    if strcmp(model.kernelType, "gaussian")
        for i = 1:noVars
            K = K + exp( -normDiff(X(:,i), X(:,i))/(2*model.params(i)^2));
        end
    elseif strcmp(model.kernelType, "matern1/2")
        for i = 1:noVars
            K   = K+ exp(-pdist2(X(:,i), X(:,i), 'cityblock')/model.params(i));
        end
    elseif strcmp(model.kernelType, "matern3/2")
        for i = 1:noVars
            PD  = pdist2(X(:,i), X(:,i), 'cityblock');
            K   = K + (1 + ((sqrt(3) * PD)/model.params(i))) .* exp(- (sqrt(3) * PD)/model.params(i));
        end
    elseif strcmp(model.kernelType, "matern5/2")
        for i = 1:noVars
            PD  = pdist2(X(:,i), X(:,i), 'cityblock');
            ND  = normDiff(X(:,i), X(:,i));
            K   = K + (1 + ((sqrt(5) * PD)/model.params(i)) + ((5 * ND)/(3*model.params(i)^2))) .* exp(- (sqrt(5) * PD)/model.params(i));
        end
    elseif strcmp(model.kernelType, "cauchy")
        for i = 1:noVars
            ND  = normDiff(X(:,i), X(:,i));
            K   = K + 1 ./ ( 1 + ( ND ./ model.params(i) ^ 2));
        end
    end

else
    if strcmp(model.kernelType, "gaussian")

        ND = normDiff(model.X, model.X);
        K  = exp(-ND/(2*(model.params(1)^2)));

    elseif strcmp(model.kernelType, "matern1/2")
        if isdlarray(model.X)
            K = exp(-pdist2(extractdata(model.X), extractdata(model.X), 'cityblock')/model.params(1));
        else
            K = exp(-pdist2(model.X, model.X, 'cityblock')/model.params(1));
        end
    elseif strcmp(model.kernelType, "matern3/2")
        if isdlarray(model.X)
            d   = pdist2(extractdata(model.X), extractdata(model.X), 'cityblock');
            K   = (1 + ((sqrt(3) * d)/model.params(1))) .* exp(- (sqrt(3) * d)/model.params(1));
        else
            d   = pdist2(model.X, model.X, 'cityblock');
            K   = (1 + ((sqrt(3) * d)/model.params(1))) .* exp(- (sqrt(3) * d)/model.params(1));
        end
    elseif strcmp(model.kernelType, "matern5/2")
        ND  = normDiff(model.X, model.X);
        if isdlarray(model.X)
            d   = pdist2(extractdata(model.X), extractdata(model.X), 'cityblock');
        else
            d   = pdist2(model.X, model.X, 'cityblock');
        end
        K   = (1 + ((sqrt(5) * d)/model.params(1)) + ((5 * ND)/(3*model.params(1)^2))) .* exp(- (sqrt(5) * d)/model.params(1));

    elseif strcmp(model.kernelType, "cauchy")
        ND  = normDiff(model.X, model.X);
        K   = 1 ./ ( 1 + ( ND ./ model.params(1) ^ 2));

    end
end

if model.center == 1
    nl      =   length(K(:,1));
    oneN    =   ones(nl,nl)/nl;
    K       =   K - oneN*K - K*oneN + oneN*K*oneN + model.params(end)*eye(length(K(:,1)));
end

end
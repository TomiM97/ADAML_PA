function model = predict(model)

if model.classification ~= 1
    model.intercept = 1;
else
    model.intercept = 0;
end

if  model.redPredict  == 1

    idxSelect   = randi(length(model.X(:,1)), 500, 1);
    newX        = model.X(idxSelect,:);
    newY        = model.Y(idxSelect, :);

    model2 = model;
    model2.x = []; model2.X = newX;
    model2.y = []; model2.Y = newY;

    KTest   = kernelRBFTest(model2);
    weights = fit(model2);
    KCal    = kernelRBF(model2);

    if model.classification ~= 1
        weights = [mean(model2.Y) - mean(KCal)*weights; weights];
    end

    [n, ~]  = size(KTest);

    if model.classification ~=1
        model.ypred = [ones(n,1), KTest] * weights;
    else
        model.ypred = KTest * weights;
    end

else

    KTest   = kernelRBFTest(model);
    weights = fit(model);
    KCal    = kernelRBF(model);

    if model.classification ~= 1
        weights = [mean(model.Y) - mean(KCal)*weights; weights];
    end

    [n, ~]  = size(KTest);

    if model.classification ~=1
        model.ypred = [ones(n,1), KTest] * weights;
    else
        model.ypred = KTest * weights;
    end

end

model.intercept = 0;
end
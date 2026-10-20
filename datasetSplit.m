function model = datasetSplit(model)

if strcmpi(model.datasetName, "PienSaimaa")
    model.Xtest = model.x;
    model.X     = model.x;

    model.Ytest = model.y;
    model.Y     = model.y;

elseif strcmpi(model.datasetName, "plant-traits-1") || strcmpi(model.datasetName, "plant-traits-2")
    model.Xtest = model.xtest;
    model.X     = model.x;

    model.Ytest = model.ytest;
    model.Y     = model.y;
elseif strcmpi(model.datasetName, "plant-independent")
    model.X     = model.x;
    model.Y     = model.y;

    load("FilteredSHIFTAll.mat");
    model.XTest = X_re;
    model.YTest = Y(:,end);

else
    [row, ~]    = size(model.x);
    c           = cvpartition(row, 'holdout', model.testSize);
    idxTraining = training(c);
    idxTest     = test(c);
    model.X     = model.x(idxTraining, :);
    model.Y     = model.y(idxTraining, :);
    model.Xtest = model.x(idxTest, :);
    model.Ytest = model.y(idxTest, :);
end


%model.YtestSD = model.SD(idxTest,:);

if strcmpi(model.datasetName, "peaks-regression")
    model.Yoriginal = model.yoriginal(idxTraining,:);
    model.Ytestoriginal = model.yoriginal(idxTest,:);

end

if model.scale == 1
    [model.Y, model.muY, model.stdY] = zscore(model.Y);
    [model.X, model.muX, model.stdX] = zscore(model.X);
    model.Xtest = normalize(model.Xtest, 'center', model.muX, 'scale', model.stdX);
    model.Ytest = normalize(model.Ytest, 'center', model.muY, 'scale', model.stdY);
end

if strcmpi(model.datasetName, "covid") || strcmpi(model.datasetName, "BeesandMites") || strcmpi(model.datasetName, "synthetic-circles-classification") || strcmpi(model.datasetName, "hyperspectral") || strcmpi(model.datasetName, "cancer")
    model.YCode     = model.ycode(idxTraining);
    model.YCodeT    = model.ycode(idxTest);
end

end
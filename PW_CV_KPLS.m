close all
clearvars
clc
addpath('functions')
addpath('common')
% Advanced data analysis and machine learning
% project



% Loading data and separating to X(input variables) and Y(response variables)
T = readtable("data_part_1.csv", ReadVariableNames=true,VariableNamingRule="preserve",ReadRowNames=true);
Y = T(:,1:20);
X = T(:,21:end);
wavelengths = str2double(X.Properties.VariableNames);
T2 = readtable("data_part_2.csv", ReadVariableNames=true,VariableNamingRule="preserve",ReadRowNames=true);

tic
% 1. Settings
numOfvariables = 20;
rNames = Y.Properties.VariableNames(:,1:numOfvariables);
Y2 = T2(:,rNames);  % only the traits that are also in data_part_1
X2 = T2(:,38:end);


% Data exploration
summary(Y);
missing_X = anymissing(X); 
missing_X2 = anymissing(X2);


% Data to arrays
Y = table2array(Y);
X = table2array(X);

Y2 = table2array(Y2);
X2 = table2array(X2);

ref = mean(X, 1)';
X = emsc_fun(X', ref, wavelengths(:))';
X2 = emsc_fun(X2', ref, wavelengths(:))';
I = ~isnan(Y);
maxLV     = 20;          % largest number of latent variables tested
Kfold     = 5;           % cross-validation folds
wl = wavelengths(:)';

% 2. Data per trait: part 1 for training, part 2 for testing

I = ~isnan(Y);
I = I & ~any(isnan(X), 2);

I2 = ~isnan(Y2);
I2 = I2 & ~any(isnan(X2), 2);

groupId = (1:size(X,1))';

x = cell(1,numOfvariables);
y = cell(1,numOfvariables);
g = cell(1,numOfvariables);

xTrain = x;
yTrain = y;
gTrain = g;
xTest = x;
yTest = y;

for k = 1:numOfvariables

    % Training data: data_part_1
    x{k} = X(I(:,k), :);
    y{k} = Y(I(:,k), k);
    g{k} = groupId(I(:,k));

    xTrain{k} = x{k};
    yTrain{k} = y{k};
    gTrain{k} = g{k};

    % Independent test data: data_part_2
    xTest{k} = X2(I2(:,k), :);
    yTest{k} = Y2(I2(:,k), k);

end

% SELECT WAVELENGTHS WITH UNBIASED CROSS-VALIDATION

numWavelengths = 20;

excluded = [951, 1321];

Q2cvMean  = nan(numOfvariables, maxLV);
R2calMean = nan(numOfvariables, maxLV);

PRESSfolds = cell(1, numOfvariables);
Q2folds    = cell(1, numOfvariables);
R2calFoldsAll = cell(1, numOfvariables);
selectedWavelengths = cell(1, numOfvariables);

for k = 1:numOfvariables

    if numel(yTrain{k}) < 3*Kfold || numel(unique(gTrain{k})) < Kfold
        warning('Trait %s has too few samples or groups. Skipping.', rNames{k});
        continue
    end

    foldId = make_folds(gTrain{k}, Kfold, 1);

    PRESS = nan(Kfold, maxLV);
    Q2 = nan(Kfold, maxLV);
    R2cal = nan(Kfold, maxLV);

    foldWavelengths = cell(Kfold, 1);

    for f = 1:Kfold

        trainIdx = foldId ~= f;
        valIdx = foldId == f;

        XtrFull = xTrain{k}(trainIdx, :);
        ytr = yTrain{k}(trainIdx);

        XvalFull = xTrain{k}(valIdx, :);
        yval = yTrain{k}(valIdx);

        if isempty(ytr) || isempty(yval)
            continue
        end

        % SELECT WAVELENGTHS USING ONLY THE CV TRAINING FOLD
        Xselect = GP_fun_t(XtrFull);
        deltaLambda = diff(wavelengths(:)');
        x_diff = diff(Xselect, 1, 2);
        validDiff = isfinite(deltaLambda) & deltaLambda > 0;
        typicalStep = median(deltaLambda(validDiff));
        validDiff = validDiff & deltaLambda <= 1.5 * typicalStep;
        validDiff(excluded(excluded <= numel(validDiff))) = false;
        validDiff(max(1, excluded(excluded > 1)-1)) = validDiff(max(1, excluded(excluded > 1)-1));
        derivative_strength = max(abs(x_diff(:, validDiff)), [], 1) ./ deltaLambda(validDiff);
        candidateIdx = find(validDiff);
        [~, order] = maxk(derivative_strength, ...
            min(numWavelengths, numel(derivative_strength)));
        selectedIdx = candidateIdx(order);
        selectedIdx = unique(selectedIdx, 'stable');
        foldWavelengths{f} = wavelengths(selectedIdx);

        % FILTER THE CV TRAINING AND VALIDATION FOLDS
        Xtr = XtrFull(:, selectedIdx);
        Xval = XvalFull(:, selectedIdx);
        nLV = min([maxLV, size(Xtr, 2), size(Xtr, 1)-1]);
        if nLV < 1
            continue
        end

        % FIT PLS AND EVALUATE EACH NUMBER OF LATENT VARIABLES

        for a = 1:nLV

            [~, ~, ~, ~, beta] = plsregress(Xtr, ytr, a);

            yhatTrain = [ones(size(Xtr,1),1), Xtr] * beta;
            yhatVal = [ones(size(Xval,1),1), Xval] * beta;

            PRESS(f,a) = sum((yval - yhatVal).^2);

            SSTval = sum((yval - mean(ytr)).^2);

            if SSTval > 0
                Q2(f,a) = 1 - PRESS(f,a) / SSTval;
            end

            SSTtrain = sum((ytr - mean(ytr)).^2);
            SSEtrain = sum((ytr - yhatTrain).^2);

            if SSTtrain > 0
                R2cal(f,a) = 1 - SSEtrain / SSTtrain;
            end

        end

    end

    PRESSfolds{k} = PRESS;
    Q2folds{k} = Q2;
    R2calFoldsAll{k} = R2cal;
    selectedWavelengths{k} = foldWavelengths;

    Q2cvMean(k,:) = mean(Q2, 1, 'omitnan');
    R2calMean(k,:) = mean(R2cal, 1, 'omitnan');

end



% 4. Number of latent variables per trait
% One-standard-error rule: smallest LV count whose Q2 is within 1 SE of the best.
nLVchosen = nan(numOfvariables, 1);
for k = 1:numOfvariables
    if isempty(Q2folds{k}), continue; end
    [Q2max, iMax] = max(Q2cvMean(k,:));
    Q2se = std(Q2folds{k}, 0, 1, 'omitnan') / sqrt(Kfold);
    nLVchosen(k) = find(Q2cvMean(k,:) >= Q2max - Q2se(iMax), 1);
    % nLVchosen(k) = iMax;                  % alternative: plain maximum
end

%% Q2 and R2 PLOTS
plot_lv_heatmap(R2calMean, rNames, ...
    'Mean R^2_{calibration} over CV folds', false, nLVchosen);

plot_lv_heatmap(Q2cvMean, rNames, ...
    'Mean Q^2_{CV} over folds (red box = maximum)', true, nLVchosen);

toc
% nLVchosen(11) = 4;                         % manual override after inspecting the heatmap

%% 5. Final K-PLS model per trait and independent testing

KPLSmodels = cell(numOfvariables, 1);
KPLSpredictions = cell(numOfvariables, 1);

KPLS_R2 = nan(numOfvariables, 1);
KPLS_RMSE = nan(numOfvariables, 1);
KPLS_MAE = nan(numOfvariables, 1);

for k = 1:numOfvariables

    fprintf('\nK-PLS: trait %d/%d: %s\n', ...
        k, numOfvariables, rNames{k});

    if isnan(nLVchosen(k))
        warning('No valid LV selection for %s. Skipping.', rNames{k});
        continue
    end

    % Retrieve training and independent test data for this trait.
    XtrainK = xTrain{k};
    YtrainK = yTrain{k};

    XtestK = xTest{k};
    YtestK = yTest{k};

    % Remove invalid training observations.
    validTrain = all(isfinite(XtrainK), 2) & isfinite(YtrainK);
    XtrainK = XtrainK(validTrain, :);
    YtrainK = YtrainK(validTrain);

    % Remove invalid test observations.
    validTest = all(isfinite(XtestK), 2) & isfinite(YtestK);
    XtestK = XtestK(validTest, :);
    YtestK = YtestK(validTest);

    if size(XtrainK, 1) < 3 || isempty(XtestK)
        warning('Insufficient training or test observations for %s.', ...
            rNames{k});
        continue
    end

    % Scale using training data only.
    muX = mean(XtrainK, 1);
    stdX = std(XtrainK, 0, 1);
    stdX(stdX == 0 | ~isfinite(stdX)) = 1;

    muY = mean(YtrainK);
    stdY = std(YtrainK);

    if stdY == 0 || ~isfinite(stdY)
        warning('Trait %s has zero response variance.', rNames{k});
        continue
    end

    XtrainScaled = (XtrainK - muX) ./ stdX;
    XtestScaled = (XtestK - muX) ./ stdX;

    YtrainScaled = (YtrainK - muY) ./ stdY;

    % Initialize model.
    model = struct();

    model.X = XtrainScaled;
    model.Y = YtrainScaled;

    model.Xtest = XtestScaled;
    model.Ytest = (YtestK - muY) ./ stdY;

    % Regression, not classification.
    model.classification = 0;
    model.regrType = 2;

    % Kernel settings.
    model.kernelType = "matern1/2";
    model.initialParam = [1, 1];
    model.params = exp(model.initialParam);

    % Additional toolbox parameters.
    model.nsamp = 1;
    model.learnRate = 0.1;
    model.dim = min(nLVchosen(k), size(XtrainScaled, 2));
    model.iter = 1000;
    model.center = 1;
    model.plot = 0;
    model.sp = 0.5;
    model.momentum = 1;
    model.family = 0;
    model.redPredict = 0;

    % Fit/predict using the current kernel parameters.
    model = predict(model);

    % Convert predictions back to the original trait scale.
    yhat = model.ypred(:) * stdY + muY;
    yobs = YtestK(:);

    if numel(yhat) ~= numel(yobs)
        error('Prediction and observation counts differ for %s.', ...
            rNames{k});
    end

    % Store results.
    KPLSmodels{k} = model;
    KPLSpredictions{k} = yhat;

    residual = yobs - yhat;

    KPLS_RMSE(k) = sqrt(mean(residual.^2));
    KPLS_MAE(k) = mean(abs(residual));

    SST = sum((yobs - mean(yobs)).^2);

    if SST > 0
        KPLS_R2(k) = 1 - sum(residual.^2) / SST;
    end

end

% Summarize independent test performance.
KPLS_summary = table( ...
    rNames(:), KPLS_R2, KPLS_RMSE, KPLS_MAE, ...
    'VariableNames', {'Trait','R2_test','RMSE_test','MAE_test'});

disp(KPLS_summary)

%% 6. Plots for one trait

k = numOfvariables; % trait to inspect (1..20)
nLV = nLVchosen(k);
yhatTest = pls_predict(BETAall{k}, xTest{k});
res = yTest{k} - yhatTest;
coef = BETAall{k}(2:end)';
coef(diff(wl) > 2*median(diff(wl))) = NaN; % break the line at removed bands

figure('Name','PLS coefficients');
plot(wl, coef, 'LineWidth', 1.2); yline(0,'k-'); grid on
xlabel('Wavelength [nm]'); ylabel('Coefficient');
title(sprintf('%s (%d LV)', rNames{k}, nLV), 'Interpreter','none');

figure('Name','LV selection');
plot(1:maxLV, Q2cvMean(k,:), '-o', 'LineWidth', 1.3); grid on; xline(nLV,'--');
xlabel('Latent variables'); ylabel('Mean Q^2_{CV}');
title(sprintf('%d-fold CV - %s', Kfold, rNames{k}), 'Interpreter','none');

figure('Name','Hold-out test','Position',[100 100 1050 350]);
subplot(1,3,1)
scatter(yTest{k}, yhatTest, 10, 'filled'); hold on
lims = [min([yTest{k}; yhatTest]) max([yTest{k}; yhatTest])];
plot(lims, lims, 'k--'); hold off; grid on; axis equal tight
xlabel('Observed'); ylabel('Predicted'); title('Observed vs predicted')
subplot(1,3,2)
scatter(yhatTest, res, 10, 'filled'); yline(0,'k-'); grid on
xlabel('Predicted'); ylabel('Residual'); title('Residuals vs predicted')
subplot(1,3,3)
histogram(res, 30); grid on
xlabel('Residual'); ylabel('Frequency'); title('Residual histogram')

%% Create measured and predicted trait matrices

% Keep samples with complete spectra in data_part_2
validX2 = ~any(isnan(X2), 2);

X2_eval = X2(validX2, :);
Ytrue = Y2(validX2, :);

numTest = size(X2_eval, 1);
Ypred = nan(numTest, numOfvariables);

% Train one PLS model per trait and predict all test samples
for k = 1:numOfvariables

    [~,~,~,~,BETA] = plsregress( ...
        xTrain{k}, yTrain{k}, nLVchosen(k));

    Ypred(:, k) = pls_predict(BETA, X2_eval);

end

% Trait names
traitNames = rNames;

% Matrices can contain NaNs in Ytrue where measurements are missing
disp(size(Ytrue))
disp(size(Ypred))

% Optional: convert the matrices to tables with trait names
YtrueTable = array2table(Ytrue, ...
    'VariableNames', traitNames);

YpredTable = array2table(Ypred, ...
    'VariableNames', traitNames);

%% Residual analysis and evaluation for all traits

Ytrue = table2array(YtrueTable);
Ypred = table2array(YpredTable);

numOfvariables = size(Ytrue, 2);
rNames = YtrueTable.Properties.VariableNames;

R2test   = nan(numOfvariables, 1);
RMSEtest = nan(numOfvariables, 1);
MAEtest  = nan(numOfvariables, 1);
MeanResidual = nan(numOfvariables, 1);
StdResidual  = nan(numOfvariables, 1);

Residuals = nan(size(Ytrue));

for k = 1:numOfvariables

    valid = ~isnan(Ytrue(:,k)) & ~isnan(Ypred(:,k));

    yTrue = Ytrue(valid,k);
    yPred = Ypred(valid,k);

    res = yTrue - yPred;
    Residuals(valid,k) = res;

    if isempty(res)
        continue;
    end

    R2test(k) = 1 - sum(res.^2) / ...
        sum((yTrue - mean(yTrue)).^2);

    RMSEtest(k) = sqrt(mean(res.^2));
    MAEtest(k) = mean(abs(res));

    MeanResidual(k) = mean(res);
    StdResidual(k) = std(res);
end

%% Summary table

summaryTable = table( ...
    rNames(:), ...
    sum(~isnan(Ytrue) & ~isnan(Ypred), 1)', ...
    R2test, RMSEtest, MAEtest, MeanResidual, StdResidual, ...
    'VariableNames', ...
    {'Trait','nTest','R2_test','RMSE_test','MAE_test', ...
     'MeanResidual','StdResidual'});

disp(summaryTable)

%% Residual plots for each trait
% now only for first

for k = 1:1

    valid = ~isnan(Ytrue(:,k)) & ~isnan(Ypred(:,k));

    yTrue = Ytrue(valid,k);
    yPred = Ypred(valid,k);
    res = yTrue - yPred;

    if isempty(res)
        continue;
    end

    figure('Name', ['Residual analysis - ', rNames{k}]);

    tiledlayout(2,2);

    % Predicted vs measured
    nexttile;
    scatter(yTrue, yPred, 'filled');
    hold on;
    limits = [min([yTrue; yPred]), max([yTrue; yPred])];
    plot(limits, limits, 'k--');
    xlabel('Measured');
    ylabel('Predicted');
    title('Measured vs predicted');
    grid on;
    axis square;

    % Residuals vs predicted
    nexttile;
    scatter(yPred, res, 'filled');
    hold on;
    yline(0, 'k--');
    xlabel('Predicted');
    ylabel('Residual (measured - predicted)');
    title('Residuals vs predicted');
    grid on;

    % Residual histogram
    nexttile;
    histogram(res);
    xlabel('Residual');
    ylabel('Frequency');
    title('Residual distribution');
    grid on;

    % Normal Q-Q plot
    nexttile;
    qqplot(res);
    title('Residual normal Q-Q plot');

    sgtitle(rNames{k}, 'Interpreter', 'none');

end


%% PCA of measured trait matrix

Y = table2array(YtrueTable);
traitNames = YtrueTable.Properties.VariableNames;

% Keep traits with at least two measured observations
nObserved = sum(isfinite(Y), 1);
validTraits = nObserved >= 2;

Y = Y(:, validTraits);
traitNamesPCA = traitNames(validTraits);

% Remove rows where all retained traits are missing
validRows = any(isfinite(Y), 2);
Y = Y(validRows, :);

% Impute missing values using the mean of each trait
for j = 1:size(Y, 2)
    missing = ~isfinite(Y(:, j));
    Y(missing, j) = mean(Y(~missing, j));
end

% Remove constant traits
traitStd = std(Y, 0, 1);
validTraits = traitStd > 0;

Y = Y(:, validTraits);
traitNamesPCA = traitNamesPCA(validTraits);

% Check that PCA can be performed
if size(Y, 1) < 2 || size(Y, 2) < 2
    error('PCA requires at least two samples and two non-constant traits.');
end

% Perform PCA
[coeff, score, latent, ~, explained, mu] = pca(Y);

% Variance explained table
PC = (1:numel(explained))';

varianceTable = table(PC, explained(:), cumsum(explained(:)), ...
    'VariableNames', ...
    {'PC','VarianceExplained_pct','CumulativeVariance_pct'});

disp(varianceTable)

% Trait loadings
loadingsTable = array2table(coeff, ...
    'VariableNames', compose('PC%d', 1:size(coeff, 2)), ...
    'RowNames', traitNamesPCA);

disp(loadingsTable)

% Plot variance explained
figure('Name', 'PCA variance explained');
tiledlayout(1, 2);

nexttile;
pareto(explained(:),0.999);
xlabel('Principal component');
ylabel('Variance explained (%)');
title('Variance explained by each PC');

nexttile;
plot(PC, cumsum(explained(:)), '-o');
xlabel('Number of principal components');
ylabel('Cumulative variance explained (%)');
title('Cumulative variance');
grid on;

% Plot sample scores
figure('Name', 'PCA scores');

scatter(score(:, 1), score(:, 2), 35, 'filled');
xlabel(sprintf('PC1 (%.1f%%)', explained(1)));
ylabel(sprintf('PC2 (%.1f%%)', explained(2)));
title('PCA score plot of measured traits');
grid on;
axis equal;

% Plot trait loading vectors
figure('Name', 'PCA trait loadings');

quiver(zeros(size(coeff, 1), 1), zeros(size(coeff, 1), 1), ...
    coeff(:, 1), coeff(:, 2), 0, 'LineWidth', 1.2);
hold on;

for k = 1:numel(traitNamesPCA)
    text(coeff(k, 1), coeff(k, 2), [' ', traitNamesPCA{k}], ...
        'Interpreter', 'none', 'FontSize', 8);
end

xlabel('PC1 loading');
ylabel('PC2 loading');
title('Trait loading vectors');
grid on;
xline(0, ':');
yline(0, ':');
axis equal;

fprintf('Number of samples used in PCA: %d\n', size(Y, 1));
fprintf('Number of traits used in PCA: %d\n', size(Y, 2));
fprintf('Number of imputed values: %d\n', sum(~isfinite(Y(:))));

%% Local functions

function yhat = pls_predict(BETA, XZ)
    % BETA(1) = intercept, BETA(2:end) = coefficients. XZ is preprocessed like the calibration data.
    yhat = [ones(size(XZ,1),1) XZ] * BETA;
end

function foldOf = make_folds(groupId, K, seed)
    % Assigns every sample to one of K folds. All samples of a group share a fold.
    g = findgroups(groupId);
    rng(seed);
    foldOfGroup = mod(randperm(max(g)), K) + 1;
    foldOf = foldOfGroup(g);
    foldOf = foldOf(:);
end

function plot_lv_heatmap(M, traitNames, titleStr, markBest, nLVchosen)
    % Trait x LV heatmap with the value printed in every cell. Colour scale fixed to 0..1.
    [nTraits, nLVmax] = size(M);
    figure('Position', [100 100 1000 600]);
    imagesc(1:nLVmax, 1:nTraits, M);
    colormap(parula); colorbar; clim([0 1]);
    xticks(1:nLVmax); yticks(1:nTraits); yticklabels(traitNames);
    set(gca, 'TickLabelInterpreter', 'none');
    xlabel('Number of latent variables'); title(titleStr);
    hold on
    for i = 1:nTraits
        for j = 1:nLVmax
            v = M(i,j);
            if isnan(v), continue; end
            if v < 0.5, c = [1 1 1]; else, c = [0 0 0]; end
            text(j, i, sprintf('%.2f', v), 'HorizontalAlignment','center', 'FontSize', 7, 'Color', c);
        end

        if markBest && ~isnan(nLVchosen(i)) && ...
                nLVchosen(i) >= 1 && nLVchosen(i) <= nLVmax
            rectangle('Position', ...
                [nLVchosen(i)-0.5, i-0.5, 1, 1], ...
                'EdgeColor','r', 'LineWidth', 1.5);
        end
    end
    hold off
end

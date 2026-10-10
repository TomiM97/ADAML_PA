close all
clearvars
clc
addpath('functions')
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

ref = mean(X, 1)';
%X = emsc_fun(X', ref, wavelengths(:))';

I = ~isnan(Y);
maxLV     = 20;          % largest number of latent variables tested
Kfold     = 5;           % cross-validation folds
testFrac  = 0.2;         % share of part 1 kept as hold-out test set
wl = wavelengths(:)';

% 2. Data per trait, grouping, hold-out split
I = I & ~any(isnan(X), 2);                  % drop rows with missing spectra

% Observations that belong together (same plant, plot, site) must stay in the same fold.
% Default: every observation is its own group. Derive groups from the row names if they encode this:
% groupId = extractBefore(T.Properties.RowNames, '_');
groupId = (1:size(X,1))';


x = cell(1,numOfvariables);  y = cell(1,numOfvariables);  g = cell(1,numOfvariables);
xTrain = x;  yTrain = y;  gTrain = g;  xTest = x;  yTest = y;
for k = 1:numOfvariables
    x{k} = X(I(:,k), :);   y{k} = Y(I(:,k), k);   g{k} = groupId(I(:,k));

    isTest = make_folds(g{k}, round(1/testFrac), 1) == 1;    % fold 1 = hold-out test set
    xTrain{k} = x{k}(~isTest,:);   yTrain{k} = y{k}(~isTest);   gTrain{k} = g{k}(~isTest);
    xTest{k}  = x{k}(isTest,:);    yTest{k}  = y{k}(isTest);
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
        validDiff(max(1, excluded(excluded > 1)-1)) = ...
            validDiff(max(1, excluded(excluded > 1)-1));
        derivative_strength = max(abs(x_diff(:, validDiff)), [], 1) ./ ...
            deltaLambda(validDiff);
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

%% 5. Final PLS model per trait and hold-out evaluation
BETAall  = cell(1, numOfvariables);          % BETAall{k}(1) = intercept, (2:end) = coefficients
R2test   = nan(numOfvariables,1);
RMSEtest = nan(numOfvariables,1);
MAEtest  = nan(numOfvariables,1);
Q2atLV   = nan(numOfvariables,1);

for k = 1:numOfvariables
    if isnan(nLVchosen(k)), continue; end
    Q2atLV(k) = Q2cvMean(k, nLVchosen(k));

    [~,~,~,~,BETA] = plsregress(xTrain{k}, yTrain{k}, nLVchosen(k));
    res = yTest{k} - pls_predict(BETA, xTest{k});
    BETAall{k} = BETA;

    R2test(k)   = 1 - sum(res.^2) / sum((yTest{k} - mean(yTest{k})).^2);
    RMSEtest(k) = sqrt(mean(res.^2));
    MAEtest(k)  = mean(abs(res));
end

summaryTable = table(rNames', cellfun(@numel,yTrain)', cellfun(@numel,yTest)', nLVchosen, ...
    Q2atLV, R2test, RMSEtest, MAEtest, 'VariableNames', ...
    {'Trait','nTrain','nTest','nLV','Q2_cv','R2_test','RMSE_test','MAE_test'});
disp(summaryTable)

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
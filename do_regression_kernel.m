% functions pls1_R2_cal.m and r2_pls1_forward_selection.m needed
close all
clearvars
clc
%% Loading data and separating to X(input variables) and Y(response variables)
T = readtable("data_part_1.csv", ReadVariableNames=true,VariableNamingRule="preserve",ReadRowNames=true);
Y = T(:,1:20);
X = T(:,21:end);
rNames = Y.Properties.VariableNames;
wavelengths = str2double(X.Properties.VariableNames);


T2 = readtable("data_part_2.csv", ReadVariableNames=true,VariableNamingRule="preserve",ReadRowNames=true);
Y2 = T2(:,rNames);  % only the traits that are also in data_part_1
X2 = T2(:,38:end);

%% Data to arrays
Y = table2array(Y);
X = table2array(X);
Y2 = table2array(Y2);
X2 = table2array(X2);



%% finding available observations
I = ~isnan(Y);    
I2 = ~isnan(Y2);

%% regression

rng(0,"twister") % For reproducibility

maxLV     = 20;          % largest number of latent variables tested
Kfold     = 5;           % cross-validation folds
doPlots = false;

[m,n] = size(Y);

tic
for i=1:n
    x = X(I(:,i), :);
    y =(Y(I(:,i),i));
    xTest = X2(I2(:,i), :);
    yTest = (Y2(I2(:,i), i));

    [ordered_by_pls1{i},ncomp(i),BETA{i},rmse_te(i),mae_te(i)] = pls1_R2_cal(x,y,xTest,yTest,Kfold,maxLV,doPlots);
end
toc
%% Kernel regression
tic

for i = 1:n

    x = X(I(:,i),:);
    y = Y(I(:,i),i);

    xTest = X2(I2(:,i),:);
    yTest = Y2(I2(:,i),i);

    [ordered_kernel{i}, ...
     kernelModel{i}, ...
     rmse_kernel(i), ...
     mae_kernel(i)] = ...
        kernel_R2_cal(x,y,xTest,yTest,Kfold,maxLV);

end

toc

%%
ref = mean(X, 1)';
X_emsc = emsc_fun(X', ref, wavelengths(:))';
X2_emsc = emsc_fun(X2', ref, wavelengths(:))';

tic
for i=1:n
    x = X_emsc(I(:,i), :);
    y = (Y(I(:,i),i));
    xTest = X2_emsc(I2(:,i), :);
    yTest = (Y2(I2(:,i), i));

    [ordered_by_pls1_emsc{i},ncomp_emsc(i),BETA_emsc{i},rmse_te_emsc(i),mae_te_emsc(i)] = pls1_R2_cal(x,y,xTest,yTest,Kfold,maxLV,doPlots);
end
toc

%% Kernel regression with EMSC

tic

for i = 1:n

    x = X_emsc(I(:,i),:);
    y = Y(I(:,i),i);

    xTest = X2_emsc(I2(:,i),:);
    yTest = Y2(I2(:,i),i);

    [ordered_kernel_emsc{i}, ...
     kernelModel_emsc{i}, ...
     rmse_kernel_emsc(i), ...
     mae_kernel_emsc(i)] = ...
        kernel_R2_cal(x,y,xTest,yTest,Kfold,maxLV);

end

toc

%% for comparison

mae = [
    mae_te'
    mae_te_emsc'
    mae_kernel'
    mae_kernel_emsc'
];

rmse = [
    rmse_te'
    rmse_te_emsc'
    rmse_kernel'
    rmse_kernel_emsc'
];


%% Predicting missing values to form a complete Y matrix
In = isnan(Y);    
In2 = isnan(Y2);
Y_full = Y;

for i=1:n
    x = X(In(:,i), :);
    x_miss = x(:,ordered_by_pls1{i});
    yhat{i} = [ones(size(x_miss,1),1) x_miss]*BETA{i};
    Y_full(In(:,i),i) = yhat{i};
end

Y_full_emsc = Y;

for i=1:n
    x_emsc = X_emsc(In(:,i), :);
    x_miss_emsc = x_emsc(:,ordered_by_pls1_emsc{i});
    yhat_emsc{i} = [ones(size(x_miss_emsc,1),1) x_miss_emsc]*BETA_emsc{i};
    Y_full_emsc(In(:,i),i) = yhat_emsc{i};
end

%% For useful comparison

modelNames = [
    "PLS"
    "PLS + EMSC"
    "Kernel"
    "Kernel + EMSC"
];

meanRMSE = mean(rmse,2,'omitnan');
meanMAE = mean(mae,2,'omitnan');

% Make all variables equal in length
modelNames = modelNames(:);
meanRMSE = meanRMSE(:);
meanMAE = meanMAE(:);

n = min([numel(modelNames), numel(meanRMSE), numel(meanMAE)]);

modelNames = modelNames(1:n);
meanRMSE = meanRMSE(1:n);
meanMAE = meanMAE(1:n);

% Create results table
results = table(modelNames, meanRMSE, meanMAE);


disp(results)

%% Predict missing Y values using kernel regression

Y_full_kernel = Y;

for i = 1:n

    xMissing = X(In(:,i),:);

    if ~isempty(xMissing)

        yPred = kernel_predict(kernelModel{i},xMissing);

        Y_full_kernel(In(:,i),i) = yPred;

    end
end

%% Predict missing Y values using kernel regression + EMSC

Y_full_kernel_emsc = Y;

for i = 1:n

    xMissing = X_emsc(In(:,i),:);

    if ~isempty(xMissing)

        yPred = kernel_predict(kernelModel_emsc{i},xMissing);

        Y_full_kernel_emsc(In(:,i),i) = yPred;

    end
end
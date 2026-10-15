% functions pls1_regression.m and r2_pls1_forward_selection.m needed

%% Loading data and separating to X(input variables) and Y(response variables)
T = readtable("data_part_1.csv", ReadVariableNames=true,VariableNamingRule="preserve",ReadRowNames=true);
Y = T(:,1:20);
X = T(:,21:end);
rNames = Y.Properties.VariableNames;
wavelenghts = str2double(X.Properties.VariableNames);


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
doPlots = true;

[m,n] = size(rNames);

for i=1:n
    x = X(I(:,i), :);
    y =(Y(I(:,i),i));
    xTest = X2(I2(:,i), :);
    yTest = (Y2(I2(:,i), i));

    [ordered_by_pls1{i},ncomp(i),rmse_te(i),mae_te(i),yhat_te{i},res_te{i}] = pls1_regression(x,y,xTest,yTest,Kfold,maxLV,doPlots);
end

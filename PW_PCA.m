close all
clearvars
clc
% Advanced data analysis and machine learning
% project



%% Loading data and separating to X(input variables) and Y(response variables)
T = readtable("data_part_1.csv", ReadVariableNames=true,VariableNamingRule="preserve",ReadRowNames=true);
Y = T(:,1:20);
X = T(:,21:end);
rNames = Y.Properties.VariableNames;
wavelenghts = str2double(X.Properties.VariableNames);



%% Data exploration
summary(Y);
missing_X = anymissing(X); 



%% Data to arrays
Y = table2array(Y);
X = table2array(X);



%% Vizualization
mi = min(X);
ma = max(X);
me = mean(X);
figure;
plot(wavelenghts,ma,'.')
hold on
plot(wavelenghts, me,'.')
plot(wavelenghts, mi,'.')
xlim([399 2501])
xlabel("Wavelength [nm]");
ylabel("Reflectance");
legend({'maximum', 'mean', 'minimum'},'FontSize',12)



%% PCA
[loadings, scores, latent, tsqrd, explained] = pca(X);

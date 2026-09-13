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

% PC1-PC2 scores
figure;
biplot(loadings(1:15:end,1:2),'Scores', scores(1:15:end,1:2))
xlabel('PC1 score');
ylabel('PC2 score');
legend('Coeff 1', 'Coeff 2', '', 'Scores')
grid on;
% PC1-PC3 loadings
figure
plot(wavelenghts, loadings(:,1:3), 'LineWidth', 1.2);
xlabel('Wavelength (nm)');
ylabel('Loading');

legend('PC1','PC2','PC3');
xlim([399 2501]);
grid on;
% Explained variance
figure;
pareto(explained);
xlabel('Principal component');
ylabel('Explained variance (%)');
grid on;
% PC2-PC3 scores
figure;
plot(scores(1:15:end,2), scores(1:15:end,3), '.');
xlabel('PC2 score');
ylabel('PC3 score');
grid on;

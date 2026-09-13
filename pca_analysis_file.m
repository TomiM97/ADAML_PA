clc; close all; clearvars

temp1 = readtable("data_part_1.csv", ReadVariableNames=true, VariableNamingRule="preserve",ReadRowNames=true);
temp2 = readtable("data_part_1.csv", ReadVariableNames=true, VariableNamingRule="preserve",ReadRowNames=true);
Y1 = temp1(:,1:20);
X1 = temp1(:,21:end);
Y2 = temp2(:,1:20);
X2 = temp2(:,21:end);
wvl = str2double(X1.Properties.VariableNames);
Y1 = table2array(Y1); X1 = table2array(X1); Y2 = table2array(Y2); X2 = table2array(X2);

plot(wvl, X1(1:3,:), '.')


%%
close all;
[loadings, scores, latent, tsqrd, explained] = pca(X1);
% PC1-PC2 scores
figure;
biplot(loadings(1:15:end,1:2),'Scores', scores(1:15:end,1:2))
xlabel('PC1 score');
ylabel('PC2 score');
legend('Coeff 1', 'Coeff 2', '', 'Scores')
grid on;
% PC1-PC3 loadings
figure
plot(wvl, loadings(:,1:3), 'LineWidth', 1.2);
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
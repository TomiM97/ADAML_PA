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

T2 = readtable("data_part_2.csv", ReadVariableNames=true,VariableNamingRule="preserve",ReadRowNames=true);
Y2 = T2(:,rNames);  % only the traits that are also in data_part_1
X2 = T2(:,38:end);



%% Data exploration
summary(Y);
missing_X = anymissing(X); 
missing_X2 = anymissing(X2);



%% Data to arrays
Y = table2array(Y);
X = table2array(X);

I = ~isnan(Y);


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

trait = 11;  % 1-20 
div = 225.23885776;

% PC1-PC2 scores
figure;
biplot(loadings(1:15:end,1:2))%,'Scores', scores(I(:,trait),1:2))
hold on
scatter(scores(I(:,trait),1)/div, scores(I(:,trait),2)/div,[],Y(I(:,trait),trait),".")
colorbar;
xlabel('PC1 score');
ylabel('PC2 score');
legend('Coeff 1', 'Coeff 2', '','Scores')
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
trait = 7;  % 1-20
figure;
scatter(scores(I(:,trait),2), scores(I(:,trait),3),[],Y(I(:,trait),trait),"filled")
xlabel('PC2 score');
ylabel('PC3 score');
title('Chlorophyll')
grid on;
colorbar;


%% separating based on trait

for i = 1:20
    x{i} = X(I(:,i),:); 
    y{i} = Y(I(:,i),i);
end




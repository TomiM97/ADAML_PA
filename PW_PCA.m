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

%%

x1 = X(I(:,1),:);   % Anthocyanin
y1 = Y(I(:,1),1);
x2 = X(I(:,2),:);   % Boron
y2 = Y(I(:,2),2);
x3 = X(I(:,3),:);   % Carbon
y3 = Y(I(:,3),3);
x4 = X(I(:,4),:);   % Calcium
y4 = Y(I(:,4),4);
x5 = X(I(:,5),:);   % Carotenoid
y5 = Y(I(:,5),5);
x6 = X(I(:,6),:);   % Cellulose
y6 = Y(I(:,6),6);
x7 = X(I(:,7),:);   % Chlorophyll
y7 = Y(I(:,7),7);
x8 = X(I(:,8),:);   % Copper
y8 = Y(I(:,8),8);
x9 = X(I(:,9),:);   % EWT
y9 = Y(I(:,9),9);
x10 = X(I(:,10),:);   % Fiber
y10 = Y(I(:,10),10);
x11 = X(I(:,11),:);   % LAI
y11 = Y(I(:,11),11);
x12 = X(I(:,12),:);   % LMA
y12 = Y(I(:,12),12);
x13 = X(I(:,13),:);   % Lignin
y13 = Y(I(:,13),13);
x14 = X(I(:,14),:);   % Magnesium
y14 = Y(I(:,14),14);
x15 = X(I(:,15),:);   % Manganese
y15 = Y(I(:,15),15);
x16 = X(I(:,16),:);   % Nitrogen
y16 = Y(I(:,16),16);
x17 = X(I(:,17),:);   % NSC
y17 = Y(I(:,17),17);
x18 = X(I(:,18),:);   % Phosphorus
y18 = Y(I(:,18),18);
x19 = X(I(:,19),:);   % Potassium
y19 = Y(I(:,19),19);
x20 = X(I(:,20),:);   % Sulfur
y20 = Y(I(:,20),20);

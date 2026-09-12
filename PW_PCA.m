clearvars;
%%
clc;
close all

T1 = load("T1.mat");
T2 = load("T2.mat");
Y1 = T1(:, 1:20);
X1 = T1(:,21:end);
Y2 = T2(:, 1:20);
X2 = T2(:,21:end);
rNames1 = Y1.Properties.VariableNames;
wavelengths1 = str2double(X1.Properties.VariableNames);
rNames2 = Y2.Properties.VariableNames;
wavelengths2 = str2double(X2.Properties.VariableNames);

Y1 = table2array(Y1);
X1 = table2array(X1);
Y2 = table2array(Y2);
X2 = table2array(X2);

plot(wavelengths1, X1(1,:), "r.")
hold on;
plot(wavelengths2, X2(1,:), "b.")

coeff1 = pca(X1);
coeff2 = pca(X2);

% PCA
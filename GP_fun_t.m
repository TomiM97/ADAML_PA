function [gpdata, matern_covs] = GP_fun_t(data)
% GAUSSIAN PROCESS
% GP ajan yli jokaiselle aallonpituudelle
% Data: (time x wvl)
[timeSize, wvlsize] = size(data);
t = (1:20:timeSize)';
t_star = linspace(0, timeSize, timeSize)';  % sama aikaväli (interpolointi)

sigma2 = 1^2;
ell = 25;
sigma_noise2 = 5e-4;
% Tallennetaan GP-ennusteet
mu_matern_all = zeros(timeSize, wvlsize);

dist = abs(t - t');
dist_star = abs(t_star - t');
I = eye(size(t,1));

% matern kernel
matern_cov = sigma2 * (1 + sqrt(3)*dist/ell) .* exp(-sqrt(3)*dist/ell);
matern_covs = sigma2 * (1 + sqrt(3)*dist_star/ell) .* exp(-sqrt(3)*dist_star/ell);
matern_noise = matern_cov + sigma_noise2 * I;
for wvl = 1:wvlsize
    y = data(t,wvl);
    % GP Prediction
    mu_matern = matern_covs * (matern_noise \ y);
    mu_matern_all(:, wvl) = mu_matern;
end
gpdata = mu_matern_all(:,:);
end
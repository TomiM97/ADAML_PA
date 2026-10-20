function [ordered_by_pls1,ncomp,rmse_te,mae_te] = pls1_regression(X,Y,Xtest,Ytest,Kfold,maxLV,doPlots)
%
%
% Inputs:
% X = X matrix of predictors for training
% Y = Y vector of response variables for training
% Xtest = X matrix of predictors for testing
% Ytest = Y vector of response variables for testing
% Kfold = number of folds for cross-validation
% maxLV = maximum number of latent variables chosen
% doPlots = draws plots if set to true
% 
%
% Outputs:
% ordered_by_pls1 = indices of X matrix columns that were chosen as the
%                   variables to be used based on the R2-method
% ncomp = number of latent variables chosen
% rmse_te = RMSE from the test set
% mae_te = MAE from the test set
% 
% 


x = X;
[y, ymu, ysigma] = zscore(Y);
xTest = Xtest;
yTest = (Ytest-ymu)/ysigma;

varNames = 1:1:1721;

[ordered_by_pls1, pls1_keep] = r2_pls1_forward_selection(x, y, varNames, 1e-3, maxLV);
x_red = x(:,pls1_keep);

xTest = xTest(:,pls1_keep);

%

nTrain  = size(x_red,1);
block   = floor(nTrain/(Kfold+1));

foldInfo = struct('calIdx',[],'valIdx',[]);
for f = 1:Kfold
    calEnd = block*f;
    valBeg = calEnd + 1;
    valEnd = min(calEnd + block, nTrain);
    if valBeg > nTrain, break; end
    foldInfo(f).calIdx = 1:calEnd;
    foldInfo(f).valIdx = valBeg:valEnd;
end
Keff = sum(~arrayfun(@(s) isempty(s.valIdx), foldInfo));

PRESS_folds = nan(Keff, maxLV);
Q2_folds    = nan(Keff, maxLV);
for a = 1:maxLV
    for f = 1:Keff
        Ci = foldInfo(f).calIdx;
        Vi = foldInfo(f).valIdx;

        Xc = x_red(Ci,:);  yc = y(Ci);
        Xv = x_red(Vi,:);  yv = y(Vi);

        % Standardize within the calibration window
        mu_f = mean(Xc,1);
        sd_f = std(Xc,0,1); sd_f(sd_f==0) = 0.001;
        XcZ  = (Xc - mu_f)./sd_f;
        XvZ  = (Xv - mu_f)./sd_f;

        r = min([a, rank(XcZ), size(XcZ,2)]);
        [~,~,~,~,BETA] = plsregress(XcZ, yc, r);

        yhat_v = [ones(size(XvZ,1),1) XvZ]*BETA;

        press_f = sum((yv - yhat_v).^2);
        ybar_cal = mean(yc);
        tss_f   = sum((yv - ybar_cal).^2);

        PRESS_folds(f,a) = press_f;
        Q2_folds(f,a)    = 1 - press_f / max(tss_f, eps);  % fold-level Q^2 (correct baseline)
    end
end

PRESS_med = mean(PRESS_folds, 1, 'omitnan');
Q2_med    = mean(Q2_folds,    1, 'omitnan');

[~, a_press_med] = min(PRESS_med);
[~, a_q2_med]    = max(Q2_med);


%

ncomp = min(max(a_press_med,a_q2_med),maxLV);   % number of LVs selected from CV plots

[XL,YL,XS,YS,BETA,PCTVAR,MSE,stats] = plsregress(x_red, y, ncomp);

yhat_te = [ones(size(xTest,1),1) xTest]*BETA;
res_te = yTest - yhat_te;


rmse_te = sqrt(mean(res_te.^2));
mae_te  = mean(abs(res_te));
%r2_te   = 1 - sum(res_te.^2)/sum((yTest - mean(yTest)).^2); % careful: this is R_test, sensitive to variance

% fprintf('Test set metrics (ncomp=%d): RMSE=%.3f, MAE=%.3f, R^2_test=%.3f\n', ...
%     ncomp, rmse_te, mae_te, r2_te);
if doPlots == 1
% Plots: mean curves (and optional fold distributions)
    % figure;
    % subplot(1,2,1)
    % plot(1:maxLV, PRESS_med, '-o','LineWidth',1.3); grid on
    % xlabel('Number of latent variables'); ylabel('mean PRESS_{CV}')
    % title(sprintf('PRESS_{CV} (mean, K=%d)', Keff));
    % xline(a_press_med,'--','Best by PRESS_{med}','LabelOrientation','horizontal','LabelVerticalAlignment','bottom');
    % 
    % subplot(1,2,2)
    % plot(1:maxLV, Q2_med, '-o','LineWidth',1.3); grid on
    % xlabel('Number of latent variables'); ylabel('mean Q^{2}_{CV}')
    % title(sprintf('Q^{2}_{CV} (mean, K=%d)', Keff));
    % ylim([min(-0.2, min(Q2_med)-0.05) 1]);
    % xline(a_q2_med,'--','Best by Q^{2}_{med}','LabelOrientation','horizontal','LabelVerticalAlignment','bottom');

    %
    figure
    tiledlayout(2,2);

    % Predicted vs measured
    nexttile;
    scatter(yTest, yhat_te, 'filled');
    hold on;
    limits = [min([yTest; yhat_te]), max([yTest; yhat_te])];
    plot(limits, limits, 'k--');
    xlabel('Measured');
    ylabel('Predicted');
    title('Measured vs predicted');
    grid on;
    axis square;

    % Residuals vs predicted
    nexttile;
    scatter(yhat_te, res_te, 'filled');
    hold on;
    yline(0, 'k--');
    xlabel('Predicted');
    ylabel('Residual (measured - predicted)');
    title('Residuals vs predicted');
    grid on;

    % Residual histogram
    nexttile;
    histogram(res_te);
    xlabel('Residual');
    ylabel('Frequency');
    title('Residual distribution');
    grid on;

    % Normal Q-Q plot
    nexttile;
    qqplot(res_te);
    title('Residual normal Q-Q plot');

    %sgtitle(rNames{k}, 'Interpreter', 'none');
    % figure('Name','Residual analysis (test set)','Position',[100 100 1050 350]);
    % subplot(1,2,1)
    % histogram(res_te, 20,'FaceColor',[0.2 0.6 0.8])
    % xlabel('Residual'); ylabel('Frequency'); grid on
    % title('Residual histogram')
    % 
    % subplot(1,2,2)
    % % Simple ACF of residuals
    % maxLag = 40;
    % res_te_c = res_te - mean(res_te);
    % acf = xcorr(res_te_c, maxLag, 'coeff');
    % stem(0:maxLag, acf(maxLag+1:end), 'filled')
    % xlabel('Lag'); ylabel('ACF'); grid on
    % title('Residual autocorrelation')
    % yline(0,'k-'); 
    % 
    % figure;
    % scatter(yTest, yhat_te, 15, 'filled'); hold on
    % plot(xlim, xlim,'k--','LineWidth',1.2); hold off
    % xlabel('Observed y (test)'); ylabel('Predicted y (test)');
    % title('Observed vs Predicted'); grid on; axis equal tight
end

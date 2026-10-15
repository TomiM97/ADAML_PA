function [ordered_vars, best_set, R2_hist] = r2_pls1_forward_selection(XZ, y, varNames, minGain, maxVars)

if istable(XZ) || istimetable(XZ), XZ = table2array(XZ); end
XZ = double(XZ);
if istable(y) || istimetable(y), y = table2array(y); end
y = double(y(:));

[n,p] = size(XZ);
if nargin < 4 || isempty(minGain), minGain = 1e-3; end
if nargin < 5 || isempty(maxVars), maxVars = p; end
if iscell(varNames), varNames = string(varNames); end

% Start: two most correlated vars with y (on train)
C = abs(corr(XZ, y, 'Rows','pairwise')); C = C(:);
[~, idx_sorted] = sort(C, 'descend');
S = idx_sorted(1:min(2,p)).';
remaining = setdiff(1:p, S);

% Helper: R^2_cal via PLS(1)
    function r2 = subset_r2_pls1(cols)
        [~,~,~,~,BETA] = plsregress(XZ(:,cols), y, 1);
        yhat = [ones(n,1) XZ(:,cols)] * BETA;
        SSE  = sum((y - yhat).^2);
        SST  = sum((y - mean(y)).^2);
        r2   = 1 - SSE / max(SST, eps);
    end

curR2 = subset_r2_pls1(S);
ordered_vars = varNames(S).';
R2_hist = curR2;

% Greedy add
while ~isempty(remaining) && numel(S) < maxVars
    bestGain = 0; bestIdx = NaN; bestR2 = curR2;
    for j = remaining
        r2_try = subset_r2_pls1([S j]);
        gain   = r2_try - curR2;
        if gain > bestGain
            bestGain = gain; bestIdx = j; bestR2 = r2_try;
        end
    end
    if isnan(bestIdx) || bestGain < minGain, break; end
    S = [S bestIdx];                              
    curR2 = bestR2;
    ordered_vars = [ordered_vars; varNames(bestIdx)]; 
    R2_hist(end+1,1) = curR2;                     
    remaining(remaining==bestIdx) = [];
end

best_set = varNames(S).';
end

function [emsc, coeffs] = emsc_fun(data, ref, wl)
    % data: wvl x nspec, each column is one spectrum
    % ref : reference spectrum (wvl x 1), leave empty to use the mean of data
    % wl  : wavelengths in nm, used for the baseline terms (gaps are handled correctly)
    wvl = size(data, 1);
    if nargin < 2 || isempty(ref), ref = mean(data, 2); end
    if nargin < 3 || isempty(wl),  wl  = (1:wvl)';       end

    lambda = 2*(wl(:) - min(wl)) / (max(wl) - min(wl)) - 1;   % normalized wavelength axis, -1..1

    % Design matrix: offset, reference, linear and quadratic baseline
    X = [ones(wvl,1), ref(:), lambda, lambda.^2];

    % Least-squares fit of all spectra at once: 4 x nspec
    B  = X \ data;
    a  = B(1,:);
    b  = B(2,:);
    d1 = B(3,:);
    d2 = B(4,:);

    % Remove baseline terms and scale by the reference coefficient
    baseline = X(:,[1 3 4]) * B([1 3 4],:);
    emsc = (data - baseline) ./ b;
    coeffs = struct('a', a, 'b', b, 'd1', d1, 'd2', d2);
end
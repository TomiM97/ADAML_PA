function nD = normDiff(x, y)

sqx     = x.^2;         sqy = y.^2;     % Square
nsx     = sum(sqx, 2);  nsy = sum(sqy, 2); % Rowwise sum
nsx     = nsx(:);       nsy = nsy(:);
innerMat= x*y';                         % Inner matrix: np.matmul(x, y')
nD      = -2 * innerMat + nsx + nsy';   % Compute norm difference

end
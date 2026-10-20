function B = kpls(K, Y, kvers, dim)
% Difference between the versions:
%       V1: Kernel Matrx deflation in regular PLS version.
%       V2: Kernel Matrix deflation in KPLS version.

if kvers == 1
    [T, P,  Q,  W,  U]  = pls(K,    Y,  dim);
    B                   = W * inv(P'*W) * Q';
elseif kvers == 2
    [~,~,~,~,BETA,~,~,~] = plsregress(K, Y, dim);
    B                    = BETA(2:end);

elseif kvers == 3
    [T, P] = pca_fun(K);
    b = T(:,1:dim)\Y;
    B = P(:,1:dim)*b;

else
    dim2       = dim;
    [~,nx]      = size(K);
    [m,ny]      = size(Y);
    T 	        = zeros(m, dim2);
    P	        = zeros(nx,dim2);
    Q	        = zeros(ny,dim2);
    W           = zeros(nx,dim2);
    U	        = zeros(m, dim2);

    for i = 1:dim2
        C = Y'*K;         % Calculate covariance matrix
        [~, D] = pca_fun(C);  % Get PCA loadings of cov. matrix
        w = D(:,1);       % initializing weights with loadings of first PC
        t = K*w;          % x-side scores
        q = Y'*t/(t'*t);  % y-side loadings
        u = Y*q/(q'*q);   % y-side scores
        T(:,i) = t;
        Q(:,i) = q;
        W(:,i) = w;
        U(:,i) = u;
        p	   = K'*t/(t'*t);
        P(:,i) = p;
%         K	   = K - t*p' - K*t*t' + t*p'*t*t';
        I      = eye(m);
        K      = (I - t*t') * K * (I-t*t');
        Y	   = Y - t*q';
    end
    B       = W * inv(P'*W) * Q';
end

end
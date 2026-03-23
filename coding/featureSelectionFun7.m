% =========================================================================
%
% Objective:
%   J(W,P) = ||B - W P'||_F^2
%          + alpha * tr(P2' * L_B * P2)     spectral smoothness
%          + beta  * ||W||_{2,1}             row sparsity
%          + gamma1 * ||W'W - I||_F^2        orthonormality
%          + gamma2 * ||B*W - P||_F^2        projection consistency
%
% B = [B1; B2]: B1 are compressed historical rows, B2 are fresh rows.
% W (D x C): feature selection matrix; row norms rank feature importance.
% P (L x C): low-dimensional embedding.
% =========================================================================
function [valueArr, alpha1, gamma1, gamma2, beta, score_sort, featureSelectIndex] = ...
        featureSelectionFun7(matrixSketch, MF_topK, shrinkLevel, ...
                             perFeature, maxIter, alpha1, gamma1, gamma2, beta)

[matrixSketchI1, matrixSketchI2] = size(matrixSketch);
eps_stab = 1e-20;   
tol      = 1e-6;    

%% Split sketch into historical (B1) and fresh (B2) parts
B1 = matrixSketch(1:shrinkLevel-1, :);       
B2 = matrixSketch(shrinkLevel:end, :);       
[L1, ~] = size(B1);
[L2, ~] = size(B2);

% Non-negative decomposition of B1 for multiplicative updates
B1_pos = max(B1,  0);   % B1⁺
B1_neg = max(-B1, 0);   % B1⁻

%% Graph Laplacian constructed on fresh data B2 only
[LB, ~, ~] = construct_Laplacian(B2);   % L2 × L2
alpha_LB   = alpha1 * LB;
M_plus     = 0.5 * (abs(alpha_LB) + alpha_LB);   
M_minus    = 0.5 * (abs(alpha_LB) - alpha_LB);  


B1pTB1p = B1_pos' * B1_pos;   % D × D
B1nTB1n = B1_neg' * B1_neg;   % D × D
B2TB2   = B2'     * B2;        % D × D

%% Initialization
W  = rand(matrixSketchI2, MF_topK);   % D × C  特征选择矩阵
P  = rand(matrixSketchI1, MF_topK);   % L × C  低维嵌入矩阵
P1 = P(1:L1, :);                      % ℓ-1 × C
P2 = P(L1+1:end, :);                  % L2 × C

%% Initialization
converge  = 0;
iter      = 1;
old_error = inf;
valueArr  = zeros(1, maxIter);

while (iter <= maxIter) && (converge == 0)

    % --- Update W ---
    % Q is the diagonal reweighting matrix for the l_{2,1} norm
    row_norms = sqrt(sum(W.^2, 2) + eps_stab);   % D × 1
    Q         = diag(0.5 ./ row_norms);            % D × D

    Num_W = (1 + gamma2) * (B1_pos' * P1 + B2' * P2) ...
            + 2 * gamma1 * W;

    Den_W = W * (P' * P) ...
            + (beta / 2) * Q * W ...
            + 2 * gamma1 * W * (W' * W) ...
            + gamma2 * (B1pTB1p + B1nTB1n + B2TB2) * W ...
            + (1 + gamma2) * (B1_neg' * P1);

    W = W .* sqrt(Num_W ./ max(Den_W, eps_stab));

    % --- Update P1 and P2 separately ---
    WtW = W' * W;   
    Num_P1 = (1 + gamma2) * (B1_pos * W);
    Den_P1 = (1 + gamma2) * P1 * WtW ...
             + (1 + gamma2) * (B1_neg * W);
    P1 = P1 .* sqrt(Num_P1 ./ max(Den_P1, eps_stab));

    Num_P2 = M_minus * P2 + (1 + gamma2) * (B2 * W);
    Den_P2 = (1 + gamma2) * P2 * WtW + M_plus * P2;
    P2 = P2 .* sqrt(Num_P2 ./ max(Den_P2, eps_stab));

    P = [P1; P2];

    % --- Compute full objective for convergence check ---
    B = [B1; B2];
    recon_err  = norm(B' - W * P', 'fro')^2;
    smooth_err = alpha1 * trace(P2' * LB * P2);
    sparse_err = beta   * sum(sqrt(sum(W.^2, 2) + eps_stab));
    orth_err   = gamma1 * norm(W' * W - eye(MF_topK), 'fro')^2;
    proj_err   = gamma2 * norm(B * W - P, 'fro')^2;

    train_error = recon_err + smooth_err + sparse_err + orth_err + proj_err;

    rel_change = abs(train_error - old_error) / (abs(old_error) + eps_stab);
    valueArr(1, iter) = rel_change;

    if rel_change < tol
        converge = 1;
    end
    old_error = train_error;
    iter = iter + 1;
end

%% Feature scoring: rank by row-wise L2 norm of W
score = sqrt(sum(W.^2, 2));   % D × 1，等价于 norm(W(i,:),2)

[score_sort, score_sort_index] = sort(score, 'descend');

num_featureSelect  = round(matrixSketchI2 * perFeature);
featureSelectIndex = score_sort_index(1:num_featureSelect);
featureSelectIndex = sort(featureSelectIndex);   

end

% -------------------------------------------------------------------------
% construct_Laplacian  —  RBF kernel graph Laplacian
% -------------------------------------------------------------------------
function [L, V, S] = construct_Laplacian(X)
n = size(X, 1);
S = kernelmatrix(X, 1);          
V = diag(sum(S, 1));              
L = V - S;                        
end


% -------------------------------------------------------------------------
% kernelmatrix  —  Gaussian (RBF) kernel matrix
%   K_ij = exp( -||xi - xj||^2 / (2*sig^2) )
% -------------------------------------------------------------------------
function K = kernelmatrix(coord, sig)
n = size(coord, 1);
K = coord * coord' / sig^2;
d = diag(K);
K = K - ones(n,1) * d' / 2;
K = K - d * ones(1,n) / 2;
K = exp(K);
end
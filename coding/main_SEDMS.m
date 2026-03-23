
clear;
% Path to the dataset CSV file (col 1: label, col 2-end: features)
% Download from: https://your-dataset-link
path = 'your path';
DataName = csvread(path);  % 第1列是标签，第2-116列是115个特征
Data = DataName(:, 2:end);
DataLabel = DataName(:, 1);
[~, numFeature] = size(Data);
DataLabel = DataLabel + 1;  % 从0-4变成1-5

%% Parameters
n           = 20000;   % total data stream length
numSample   = 2000;    % sketch size L
shrinkLevel = 10;      % FD retained rows after shrink (ell)
MF_topK     = 6;       % latent dimension C
perFeature  = 0.4;     % feature selection ratio
maxIter     = 20;      % max optimization iterations

alpha1  = 5000;
beta    = 10;
gamma1  = 1e4;
gamma2  = 1e4;

%% Error-trigger parameters
eta   = 1e6;    


%% Sketch initialization
matrixSketchI1 = numSample;
matrixSketchI2 = numFeature;
matrixSketch   = zeros(matrixSketchI1, matrixSketchI2);
topk           = min(matrixSketchI1, matrixSketchI2);

indexZore = 1;    % row write pointer
e         = 0;    % accumulated discarded energy
E         = 0;    % accumulated data energy
delta     = 1 / 2*matrixSketchI1;   

%% Result buffers
maxBatches     = ceil(n / (matrixSketchI1 - shrinkLevel + 1)) + 10;
ValueArr_ACC    = zeros(1, maxBatches);
ValueArr_NMI    = zeros(1, maxBatches);
ValueArr_FMeasur= zeros(1, maxBatches);
ValueArr_count  = 0;
TimeArr         = zeros(1, maxBatches);
TimeArrCount    = 1;

%% Streaming loop
for c = 1:n
    x_t = Data(c, :);

    matrixSketch(indexZore, :) = x_t;
    indexZore = indexZore + 1;

   
    E = E + norm(x_t, 2)^2;

    trigger      = false;
    triggerReason = '';

    % Trigger-1: capacity full
    if indexZore > matrixSketchI1   
        trigger       = true;
        triggerReason = 'Trigger-1 (capacity)';
    elseif (E > 0) && (e / E > delta)  % Trigger-2: error ratio exceede
        trigger       = true;
        triggerReason = 'Trigger-2 (error)';
    end

    if trigger
        % (1) Feature selection
        tStart1 = tic;
        [valueArr, alpha1, gamma1, gamma2, beta, ~, featureSelectIndex] = ...
            featureSelectionFun7(matrixSketch, MF_topK, shrinkLevel, ...
            perFeature, maxIter, alpha1, gamma1, gamma2, beta);
        time1 = toc(tStart1);

        % (2) Clustering and evaluation on selected features
        feaSelection_data = Data(1:c, featureSelectIndex);
        numK  = max(DataLabel(1:c, :));
        label = litekmeans(feaSelection_data, numK, 'MaxIter', 50, 'Replicates', 100);

        num_current_clusters = length(unique(label));
        [acc, nmi, purity, fmeasure, ~, ~] = ...
            calculate_dynamic_clustering_results(label, DataLabel(1:c, :), num_current_clusters);

        ValueArr_count = ValueArr_count + 1;
        ValueArr_ACC(1, ValueArr_count)     = acc;
        ValueArr_NMI(1, ValueArr_count)     = nmi;
        ValueArr_FMeasur(1, ValueArr_count) = fmeasure;

        fprintf('  acc=%.4f, nmi=%.4f, purity=%.4f, fmeasure=%.4f\n', acc, nmi, purity, fmeasure);

        % (3) FD shrink: compress sketch and accumulate discarded energy
        tStart2 = tic;
        [matrixSketch, indexZore, sigma_ell_sq] = ...
            spaceReleasing_SEDMS(matrixSketch, topk, shrinkLevel, matrixSketchI1);
        time2 = toc(tStart2);

        
        e = e + sigma_ell_sq;
        totalTime = time1 + time2;
        TimeArr(1, TimeArrCount) = totalTime;
        TimeArrCount = TimeArrCount + 1;
    end
end

%% Summary
if ValueArr_count > 0
    fprintf('\n统计数值: aver_ACC=%.4f, NMI=%.4f, fmeasure=%.4f\n', ...
        sum(ValueArr_ACC(1, 1:ValueArr_count))  / ValueArr_count, ...
        sum(ValueArr_NMI(1, 1:ValueArr_count))  / ValueArr_count, ...
        sum(ValueArr_FMeasur(1,1:ValueArr_count))/ ValueArr_count);
end

sumTime = sum(TimeArr(1, 1:TimeArrCount-1));
fprintf('总耗时: %.4f 秒\n', sumTime);

 



% =========================================================================
% spaceReleasing_SEDMS  —  Frequent Directions sketch compression
%
% Performs truncated SVD, applies the shrink operator, and reconstructs
% the sketch. Returns sigma_ell_sq for Trigger-2 energy tracking.
% ========================================================================
function [matrixSketch, indexZore, sigma_ell_sq] =spaceReleasing_SEDMS(matrixSketch, topk, shrinkLevel, matrixSketch_I1)
[~, S, V] = svds(matrixSketch, topk);
sigma_ell_sq = S(shrinkLevel, shrinkLevel)^2;
[newS] = shrinkOperationRow(S, shrinkLevel, topk, matrixSketch_I1);
matrixSketch = newS * V';
indexZore = shrinkLevel;
end


% -------------------------------------------------------------------------
% shrinkOperationRow  —  soft-threshold singular values
%
%   sigma_tilde_i = sqrt( max(sigma_i^2 - sigma_ell^2, 0) )
% -------------------------------------------------------------------------
function [newS] = shrinkOperationRow(S, L, topK, matrixSketchI1)
if L > topK
    error('myComponent:inputError', ...
        'shrinkLevel(%d) > topK(%d)，参数设置有误。', L, topK);
end
for i = 1:topK
    S(i, i) = S(i, i)^2;
end
value = S(L, L);
cutS = zeros(topK, topK);
for i = 1:topK
    cutS(i, i) = value;
end
temp         = max(S - cutS, 0);
afterShrinkS = sqrt(temp);

if matrixSketchI1 > topK
    newS = [afterShrinkS; zeros(matrixSketchI1 - topK, topK)];
elseif matrixSketchI1 == topK
    newS = afterShrinkS;
else
    error('myComponent:inputError', ...
        'matrixSketchI1(%d) < topK(%d)，参数设置有误。', matrixSketchI1, topK);
end
end
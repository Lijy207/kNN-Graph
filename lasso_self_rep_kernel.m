function [W, K, sigma, lambda2_adaptive, adaptive_weights] = lasso_self_rep_kernel(X, params)
% LASSO_SELF_REP_KERNEL  Kernelized self-representation with L1 sparsity
%  每列非零元素个数控制在1-5之间，并实现自适应稀疏度控制、基于类别的自适应、加权多尺度密度估计和自适应权重选择
%
%   [W, K, sigma, lambda2_adaptive, adaptive_weights] = lasso_self_rep_kernel(X, params)
%
% 输入:
%   X      - d x n 矩阵（每列为样本；若 X 为 n x d，请先转置）
%   params - 包含所有参数的结构体，字段包括：
%            lambda2: 基准正则化系数 (L1 penalty)，默认 1
%            sigma: 高斯核带宽；若传入 []，则采用中位数启发式，默认 []
%            K_keep: [] 表示用 L1 稀疏化；若为正整数，则每列仅保留 |w| 最大的 K_keep 个，默认 []
%            maxIter: 最大迭代次数，默认 200
%            tol: 收敛阈值，默认 1e-5
%            k_density_values: 用于多尺度密度估计的近邻数数组，默认 [3, 7, 15]
%            weights: 各尺度密度估计的权重数组，默认 []（表示使用自适应权重）
%            lambda2_min: 最小正则化系数，默认 lambda2 * 0.1
%            lambda2_max: 最大正则化系数，默认 lambda2 * 10
%            Y: n x 1 类别标签向量，默认 ones(size(X, 2), 1)
%
% 输出:
%   W               - n x n 稀疏系数矩阵，每列非零元素个数在1-5之间
%   K               - n x n 高斯核矩阵
%   sigma           - 实际使用的高斯核带宽
%   lambda2_adaptive- 每个样本使用的自适应正则化系数
%   adaptive_weights- 自适应选择的权重数组
%
% -------------------------------------------------------------------------
% 示例:
%   X = rand(50, 100); % 50维，100个样本
%   Y = randi(3, 100, 1); % 3个类别
%   params.lambda2 = 0.1;
%   params.k_density_values = [3, 7, 15]; % 多尺度密度估计
%   params.Y = Y;
%   [W, K, sigma, lambda2_adaptive, adaptive_weights] = lasso_self_rep_kernel(X, params);
%   spy(W); title('稀疏系数矩阵 W');
% -------------------------------------------------------------------------

    % 设置默认参数
    defaultParams.lambda2 = 1;
    defaultParams.sigma = [];
    defaultParams.K_keep = [];
    defaultParams.maxIter = 5;
    defaultParams.tol = 1e-5;
    defaultParams.k_density_values = [3, 7, 15];
    defaultParams.weights = [];
    defaultParams.Y = ones(size(X, 2), 1);
    
    % 合并参数
    if nargin < 2
        params = defaultParams;
    else
        params = mergeParams(defaultParams, params);
    end
    
    % 提取参数
    lambda2 = params.lambda2;
    sigma = params.sigma;
    K_keep = params.K_keep;
    maxIter = params.maxIter;
    tol = params.tol;
    k_density_values = params.k_density_values;
    weights = params.weights;
    Y = params.Y;
    
    % 确保Y是列向量
    Y = Y(:);
    
    % 设置 lambda2_min 和 lambda2_max 的默认值
    if isfield(params, 'lambda2_min') && ~isempty(params.lambda2_min)
        lambda2_min = params.lambda2_min;
    else
        lambda2_min = lambda2 * 0.1;
    end
    
    if isfield(params, 'lambda2_max') && ~isempty(params.lambda2_max)
        lambda2_max = params.lambda2_max;
    else
        lambda2_max = lambda2 * 10;
    end

    [d, n] = size(X);

    % ===== 高斯核矩阵 =====
    XX = sum(X.^2, 1);
    D2 = bsxfun(@plus, XX', XX) - 2*(X'*X);     % n x n
    D2(D2 < 0) = 0;                             % 数值稳定

    if isempty(sigma)
        upperTri = D2(triu(true(n),1));
        med = median(sqrt(upperTri));
        if med <= 0, med = 1; end
        sigma = med;
    end
    K = exp(-D2/(2*sigma^2));                   % n x n

    % ===== 计算基于类别的相似性矩阵 =====
    % 向量化计算类别相似性矩阵
    K_class = 0.1 * ones(n, n);
    K_class(Y == Y') = 1;  % 利用广播比较
    
    % ===== 结合高斯核相似性和类别相似性 =====
    % 使用加权组合，可以根据需要调整权重
    alpha = 0.7; % 高斯核权重
    beta = 0.3;  % 类别相似性权重
    K_combined = alpha * K + beta * K_class;

    % ===== 自适应权重选择 =====
    if isempty(weights)
        % 如果没有提供权重，使用自适应权重选择
        weights = adaptive_weight_selection(X, Y, k_density_values, D2);
    end
    
    % 检查权重数组是否与尺度数组长度一致
    if length(weights) ~= length(k_density_values)
        error('权重数组的长度必须与尺度数组的长度相同');
    end
    
    % 归一化权重，使其总和为1
    weights = weights / sum(weights);
    
    % 保存自适应选择的权重
    adaptive_weights = weights;

    % ===== 加权多尺度密度估计 =====
    densities = zeros(n, 1);
    num_scales = length(k_density_values);
    
    % 预计算类别索引
    uniqueY = unique(Y);
    classIndices = cell(length(uniqueY), 1);
    for c = 1:length(uniqueY)
        classIndices{c} = find(Y == uniqueY(c));
    end
    
    % 为每个尺度计算密度（将parfor改为for）
    for scale_idx = 1:num_scales
        k_density = k_density_values(scale_idx);
        scale_weight = weights(scale_idx);
        scale_densities = zeros(n, 1);
        
        % 非并行计算当前尺度下的密度（将parfor改为for）
        for j = 1:n
            % 找到当前样本的类别索引
            c = find(uniqueY == Y(j), 1);
            same_class_idx = classIndices{c};
            
            if length(same_class_idx) > 1 % 确保有同类样本
                % 计算到同类样本的距离
                same_class_dists = D2(j, same_class_idx);
                same_class_dists(same_class_dists == 0) = []; % 移除自身（如果存在）
                
                if ~isempty(same_class_dists)
                    k_val = min(k_density, length(same_class_dists));
                    sorted_d = sort(same_class_dists);
                    avg_dist = mean(sorted_d(1:k_val));
                    scale_densities(j) = 1 / (avg_dist + eps); % 距离越小，密度越大
                else
                    % 如果没有同类样本（除了自己），使用全局密度
                    all_dists = D2(j, :);
                    all_dists(j) = []; % 移除自身
                    k_val = min(k_density, length(all_dists));
                    sorted_d = sort(all_dists);
                    avg_dist = mean(sorted_d(1:k_val));
                    scale_densities(j) = 1 / (avg_dist + eps);
                end
            else
                % 如果没有同类样本，使用全局密度
                all_dists = D2(j, :);
                all_dists(j) = []; % 移除自身
                k_val = min(k_density, length(all_dists));
                sorted_d = sort(all_dists);
                avg_dist = mean(sorted_d(1:k_val));
                scale_densities(j) = 1 / (avg_dist + eps);
            end
        end
        
        % 归一化当前尺度的密度
        min_scale_density = min(scale_densities);
        max_scale_density = max(scale_densities);
        if max_scale_density > min_scale_density
            scale_densities_normalized = (scale_densities - min_scale_density) / (max_scale_density - min_scale_density);
        else
            scale_densities_normalized = ones(n, 1);
        end
        
        % 加权累加当前尺度的密度
        densities = densities + scale_weight * scale_densities_normalized;
    end
    
    % ===== 计算自适应正则化系数 =====
    lambda2_adaptive = zeros(n, 1);
    
    % 根据密度计算自适应正则化系数
    % 密度高的区域使用较小的lambda2（更少的正则化，更多的连接）
    % 密度低的区域使用较大的lambda2（更强的正则化，更少的连接）
    for j = 1:n
        lambda2_adaptive(j) = lambda2_min + (lambda2_max - lambda2_min) * (1 - densities(j));
    end

    % ===== 预计算 =====
    A = K_combined; % 使用结合后的相似性矩阵
    L = sum(A.^2, 1)';                          % 每列范数平方
    W = sparse(n, n); % 使用稀疏矩阵存储，因为W最终会很稀疏

    % ===== 坐标下降（逐列）- 非并行化（将parfor改为for） =====
    for j = 1:n
        y = A(:, j);
        w = zeros(n,1);
        w(j) = 0;
        r = y;  % 残差
        
        % 使用自适应正则化系数
        lambda2_j = lambda2_adaptive(j);

        for it = 1:maxIter
            w_old = w;
            for i = 1:n
                if i == j, continue; end
                ai = A(:, i);
                rho = ai' * r + L(i) * w(i);
                w_new_i = soft_threshold(rho, lambda2_j) / L(i);

                if w_new_i ~= w(i)
                    r = r - ai * (w_new_i - w(i));
                    w(i) = w_new_i;
                end
            end
            if norm(w - w_old, 2) < tol * max(1.0, norm(w_old,2))
                break;
            end
        end
        
% === 按照提供的代码形式进行稀疏化处理 ===
% 首先，移除自身（如果存在）
w(j) = 0;

% 找到非零元素
non_zero_mask = w ~= 0;

if any(non_zero_mask)
    non_zero_vals = abs(w(non_zero_mask));
    
    mean_val = mean(non_zero_vals);
    std_val = std(non_zero_vals);
    max_val = max(non_zero_vals);
    
    % 使用列索引和固定种子生成多样性因子
    col_factor = mod(j * 7 + 13, 10) / 10; 
    
    % 根据统计特性决定目标K值（1-3之间）
    if std_val > 0.5 * mean_val
        % 高方差区域：系数差异大，保留较多连接（2-3个）
        k_target = max(1, min(3, round(2 + 1 * col_factor))); 
    elseif std_val < 0.1 * mean_val
        % 低方差区域：系数差异小，保留较少连接（1-2个）
        k_target = max(1, min(3, round(1 + 1 * col_factor))); 
    else
        % 中等方差区域：均衡选择（1-3个）
        k_target = max(1, min(3, round(1 + 2 * col_factor)));
    end
    
    % 添加随机扰动增加多样性
    if rand() > 0.7  % 30%的概率添加扰动
        k_target = max(1, min(3, k_target + randi([-1,1])));
    end
    
    % 确保k_target不超过实际非零元素数量
    k_target = min(k_target, sum(non_zero_mask));
    
    % 保留前k_target个最大的非零元素
    [sorted_vals, sorted_idx] = sort(abs(w), 'descend');
    w_new = zeros(size(w));
    w_new(sorted_idx(1:k_target)) = w(sorted_idx(1:k_target));
    
    w = w_new;
else
    % 如果没有非零元素，强制选择1-2个同类样本
    same_class_idx = find(Y == Y(j));
    same_class_idx(same_class_idx == j) = []; % 移除自身
    
    col_factor = mod(j * 7 + 13, 10) / 10;
    k_target = max(1, min(2, round(1 + 1 * col_factor))); 
    
    if ~isempty(same_class_idx)
        % 选择同类样本中相似度最高的k_target个
        similarities = K_combined(same_class_idx, j);
        [~, sorted_idx] = sort(similarities, 'descend');
        selected_idx = same_class_idx(sorted_idx(1:min(k_target, length(sorted_idx))));
        w(selected_idx) = 1; % 设置为1表示连接
    else
        % 如果没有同类样本，选择所有样本中相似度最高的
        [~, idxMax] = max(K_combined(:, j));
        if idxMax == j
            % 极端情况：自己是最大值，则选第二大
            [~, ord] = sort(K_combined(:, j), 'descend');
            idxMax = ord(2);
        end
        w(idxMax) = 1;
    end
end

% 将列向量w存入稀疏矩阵W的第j列
W(:, j) = sparse(w);
    end
    
    % 输出统计信息以验证多样性
    % nnz_per_column = full(sum(W ~= 0, 1));
    % fprintf('W矩阵每列非零元素统计:\n');
    % for k = 1:5
    %     count = sum(nnz_per_column == k);
    %     fprintf('  有%d个非零元素的列数: %d (%.1f%%)\n', k, count, count/n*100);
    % end
end

% ===== 参数合并函数 =====
function merged = mergeParams(default, user)
    % 合并参数结构体
    merged = default;
    userFields = fieldnames(user);
    for i = 1:length(userFields)
        field = userFields{i};
        if isfield(merged, field)
            merged.(field) = user.(field);
        end
    end
end

% ===== 自适应权重选择函数 =====
function weights = adaptive_weight_selection(X, Y, k_density_values, D2)
    % 根据数据特性自适应选择权重
    
    n = size(X, 2); % 注意：X 是 d x n 矩阵
    num_scales = length(k_density_values);
    
    % 计算全局密度指标
    global_densities = zeros(num_scales, 1);
    for scale_idx = 1:num_scales
        k_density = k_density_values(scale_idx);
        scale_densities = zeros(n, 1);
        
        % 预计算类别索引
        uniqueY = unique(Y);
        classIndices = cell(length(uniqueY), 1);
        for c = 1:length(uniqueY)
            classIndices{c} = find(Y == uniqueY(c));
        end
        
        % 非并行计算当前尺度的密度（将parfor改为for）
        for j = 1:n
            % 找到当前样本的类别索引
            c = find(uniqueY == Y(j), 1);
            same_class_idx = classIndices{c};
            
            if length(same_class_idx) > 1
                % 计算到同类样本的距离
                same_class_dists = D2(j, same_class_idx);
                same_class_dists(same_class_dists == 0) = [];
                
                if ~isempty(same_class_dists)
                    k_val = min(k_density, length(same_class_dists));
                    sorted_d = sort(same_class_dists);
                    avg_dist = mean(sorted_d(1:k_val));
                    scale_densities(j) = 1 / (avg_dist + eps);
                else
                    all_dists = D2(j, :);
                    all_dists(j) = [];
                    k_val = min(k_density, length(all_dists));
                    sorted_d = sort(all_dists);
                    avg_dist = mean(sorted_d(1:k_val));
                    scale_densities(j) = 1 / (avg_dist + eps);
                end
            else
                all_dists = D2(j, :);
                all_dists(j) = [];
                k_val = min(k_density, length(all_dists));
                sorted_d = sort(all_dists);
                avg_dist = mean(sorted_d(1:k_val));
                scale_densities(j) = 1 / (avg_dist + eps);
            end
        end
        
        % 计算当前尺度的平均密度
        global_densities(scale_idx) = mean(scale_densities);
    end
    
    % 归一化全局密度
    min_global_density = min(global_densities);
    max_global_density = max(global_densities);
    if max_global_density > min_global_density
        global_densities_normalized = (global_densities - min_global_density) / (max_global_density - min_global_density);
    else
        global_densities_normalized = ones(num_scales, 1);
    end
    
    % 根据全局密度选择权重
    % 密度高的尺度（较小尺度）通常能更好地捕捉局部结构，给予较高权重
    % 密度低的尺度（较大尺度）通常能更好地捕捉全局结构，给予较高权重
    
    % 计算密度变化的斜率
    if num_scales > 1
        density_slope = diff(global_densities_normalized);
        avg_slope = mean(density_slope);
        
        if avg_slope > 0.1
            % 密度随尺度增加而增加，说明数据更倾向于全局结构
            weights = linspace(0.2, 0.5, num_scales); % 较大尺度权重更高
        elseif avg_slope < -0.1
            % 密度随尺度增加而减少，说明数据更倾向于局部结构
            weights = linspace(0.5, 0.2, num_scales); % 较小尺度权重更高
        else
            % 密度变化平缓，使用均衡权重
            weights = ones(1, num_scales) / num_scales;
        end
    else
        % 只有一个尺度，权重为1
        weights = 1;
    end
    
    % 确保权重总和为1
    weights = weights / sum(weights);
end

% ===== 软阈值函数 =====
function z = soft_threshold(x, t)
    z = sign(x) .* max(abs(x) - t, 0);
end
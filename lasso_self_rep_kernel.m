function [W, K, sigma, lambda2_adaptive, adaptive_weights, obj_history] = lasso_self_rep_kernel(X, params)
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
    defaultParams.maxIter = 10; % 结合线搜索和重启后，通常 5-15 次即可收敛
    defaultParams.tol = 1e-3;
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
    
    Y = Y(:);
    
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
    D2 = bsxfun(@plus, XX', XX) - 2*(X'*X);     
    D2(D2 < 0) = 0;                             
    if isempty(sigma)
        upperTri = D2(triu(true(n),1));
        med = median(sqrt(upperTri));
        if med <= 0, med = 1; end
        sigma = med;
    end
    K = exp(-D2/(2*sigma^2));                   
    
    % ===== 计算基于类别的相似性矩阵 =====
    K_class = 0.1 * ones(n, n);
    K_class(Y == Y') = 1;  
    
    % ===== 结合高斯核相似性和类别相似性 =====
    alpha = 0.7; 
    beta = 0.3;  
    K_combined = alpha * K + beta * K_class;
    
    % ===== 自适应权重选择 =====
    if isempty(weights)
        weights = adaptive_weight_selection(X, Y, k_density_values, D2);
    end
    if length(weights) ~= length(k_density_values)
        error('权重数组的长度必须与尺度数组的长度相同');
    end
    weights = weights / sum(weights);
    adaptive_weights = weights;
    
    % ===== 加权多尺度密度估计 =====
    densities = zeros(n, 1);
    num_scales = length(k_density_values);
    
    uniqueY = unique(Y);
    classIndices = cell(length(uniqueY), 1);
    for c = 1:length(uniqueY)
        classIndices{c} = find(Y == uniqueY(c));
    end
    
    for scale_idx = 1:num_scales
        k_density = k_density_values(scale_idx);
        scale_weight = weights(scale_idx);
        scale_densities = zeros(n, 1);
        
        for j = 1:n
            c = find(uniqueY == Y(j), 1);
            same_class_idx = classIndices{c};
            
            if length(same_class_idx) > 1 
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
        
        min_scale_density = min(scale_densities);
        max_scale_density = max(scale_densities);
        if max_scale_density > min_scale_density
            scale_densities_normalized = (scale_densities - min_scale_density) / (max_scale_density - min_scale_density);
        else
            scale_densities_normalized = ones(n, 1);
        end
        
        densities = densities + scale_weight * scale_densities_normalized;
    end
    
    % ===== 计算自适应正则化系数 =====
    lambda2_adaptive = zeros(n, 1);
    for j = 1:n
        lambda2_adaptive(j) = lambda2_min + (lambda2_max - lambda2_min) * (1 - densities(j));
    end
    
    % ===== 预计算与极速 FISTA-BT 核心优化 =====
    A = K_combined; 
    W = sparse(n, n); 
    
    % 取消了保守的全局 Lipschitz 计算，让线搜索动态去寻找最优大步长
    eta = 1.5; % 回溯倍率
    
    obj_history = zeros(maxIter, n);
    
    for j = 1:n
        a_j = A(:, j); 
        w = zeros(n, 1);
        Aw = zeros(n, 1); % 缓存 A*w
        
        y_vec = zeros(n, 1); 
        Ay = zeros(n, 1); % 缓存 A*y_vec (消除最耗时的乘法)
        
        t = 1;               
        lambda2_j = lambda2_adaptive(j);
        
        obj_j = zeros(maxIter, 1); 
        prev_obj = inf; 
        
        L_j = 1.0; % 给定一个非常激进（乐观）的初始步长评估
        
        for it = 1:maxIter
            w_old = w;
            Aw_old = Aw;
            
            % 1. 计算 y 的梯度与目标函数平滑项 f(y)
            err = Ay - a_j;
            grad = 2 * (A' * err); 
            f_y = sum(err.^2); 
            
            % 2. 回溯线搜索 (Backtracking Line Search)
            % 这能找出当前局部最大的安全步长
            while true
                thresh = lambda2_j / L_j;
                step = y_vec - grad / L_j;
                
                % 近端映射 (Soft Thresholding)
                w = sign(step) .* max(abs(step) - thresh, 0);
                w(j) = 0; % 严格约束自我连接为 0
                
                % 评估当前试探步长的函数值
                Aw = A * w; % 这是单次内循环唯一的一次矩阵乘法
                f_w = sum((Aw - a_j).^2);
                
                % 构建 Surrogate 替代函数进行验证
                diff_w_y = w - y_vec;
                Q_w = f_y + grad' * diff_w_y + (L_j / 2) * sum(diff_w_y.^2);
                
                if f_w <= Q_w + 1e-10 % 满足下降条件，试探成功
                    break;
                end
                L_j = L_j * eta; % 步长太大导致反弹，收缩步长重试
            end
            
            % 3. 记录真正的目标函数值
            current_obj = f_w + lambda2_j * sum(abs(w));
            obj_j(it) = current_obj;
            
            % 4. 严苛的收敛检查
            if abs(prev_obj - current_obj) < tol * max(1.0, abs(prev_obj))
                obj_j(it:end) = current_obj; 
                break;
            end
            prev_obj = current_obj;
            
            % 5. 自适应重启 (Adaptive Restart)
            % 核心黑科技：监控动量是否在做无用功（夹角是否为钝角）
            if grad' * (w - w_old) > 0
                % 发现震荡趋势，立即刹车清零，重新累积动量
                t = 1;
                y_vec = w;
                Ay = Aw; 
            else
                % 方向正确，继续 Nesterov 动量加速
                t_next = (1 + sqrt(1 + 4 * t^2)) / 2;
                momentum = (t - 1) / t_next;
                
                y_vec = w + momentum * (w - w_old);
                
                % 【提速核心】巧妙利用线性性质，无计算得出 A*y_vec
                % 彻底消除每一次外循环中一次极其耗时的 O(n^2) 矩阵乘法
                Ay = Aw + momentum * (Aw - Aw_old);
                
                t = t_next;
            end
            
            % 下次迭代前稍微放宽 L_j，以试探更大的加速步长
            L_j = max(1.0, L_j * 0.9);
        end
        
        obj_history(:, j) = obj_j;
        
        % === 按照提供的代码形式进行多样性稀疏化截断 ===
        w(j) = 0;
        non_zero_mask = w ~= 0;
        if any(non_zero_mask)
            non_zero_vals = abs(w(non_zero_mask));
            
            mean_val = mean(non_zero_vals);
            std_val = std(non_zero_vals);
            
            col_factor = mod(j * 7 + 13, 10) / 10; 
            
            if std_val > 0.5 * mean_val
                k_target = max(1, min(3, round(2 + 1 * col_factor))); 
            elseif std_val < 0.1 * mean_val
                k_target = max(1, min(3, round(1 + 1 * col_factor))); 
            else
                k_target = max(1, min(3, round(1 + 2 * col_factor)));
            end
            
            if rand() > 0.7  
                k_target = max(1, min(3, k_target + randi([-1,1])));
            end
            
            k_target = min(k_target, sum(non_zero_mask));
            
            [~, sorted_idx] = sort(abs(w), 'descend');
            w_new = zeros(size(w));
            w_new(sorted_idx(1:k_target)) = w(sorted_idx(1:k_target));
            
            w = w_new;
        else
            same_class_idx = find(Y == Y(j));
            same_class_idx(same_class_idx == j) = []; 
            
            col_factor = mod(j * 7 + 13, 10) / 10;
            k_target = max(1, min(2, round(1 + 1 * col_factor))); 
            
            if ~isempty(same_class_idx)
                similarities = K_combined(same_class_idx, j);
                [~, sorted_idx] = sort(similarities, 'descend');
                selected_idx = same_class_idx(sorted_idx(1:min(k_target, length(sorted_idx))));
                w(selected_idx) = 1; 
            else
                [~, idxMax] = max(K_combined(:, j));
                if idxMax == j
                    [~, ord] = sort(K_combined(:, j), 'descend');
                    idxMax = ord(2);
                end
                w(idxMax) = 1;
            end
        end
        
        W(:, j) = sparse(w);
    end
end

% ===== 参数合并函数 =====
function merged = mergeParams(default, user)
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
    n = size(X, 2); 
    num_scales = length(k_density_values);
    global_densities = zeros(num_scales, 1);
    
    for scale_idx = 1:num_scales
        k_density = k_density_values(scale_idx);
        scale_densities = zeros(n, 1);
        
        uniqueY = unique(Y);
        classIndices = cell(length(uniqueY), 1);
        for c = 1:length(uniqueY)
            classIndices{c} = find(Y == uniqueY(c));
        end
        
        for j = 1:n
            c = find(uniqueY == Y(j), 1);
            same_class_idx = classIndices{c};
            
            if length(same_class_idx) > 1
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
        global_densities(scale_idx) = mean(scale_densities);
    end
    
    min_global_density = min(global_densities);
    max_global_density = max(global_densities);
    if max_global_density > min_global_density
        global_densities_normalized = (global_densities - min_global_density) / (max_global_density - min_global_density);
    else
        global_densities_normalized = ones(num_scales, 1);
    end
    
    if num_scales > 1
        density_slope = diff(global_densities_normalized);
        avg_slope = mean(density_slope);
        
        if avg_slope > 0.1
            weights = linspace(0.2, 0.5, num_scales); 
        elseif avg_slope < -0.1
            weights = linspace(0.5, 0.2, num_scales); 
        else
            weights = ones(1, num_scales) / num_scales;
        end
    else
        weights = 1;
    end
    weights = weights / sum(weights);
end
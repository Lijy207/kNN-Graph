% 设置基础随机种子以确保结果可重现
rng(5, 'twister'); % 使用5或15作为基础随机种子
% 载入数据
clear; clc;
load('binalpha_uni.mat'); % X: n x d, Y: n x 1

% 将稀疏矩阵转为稠密矩阵以防止距离计算报错
if issparse(X)
    X = full(X); 
end
if issparse(Y)
    Y = full(Y);
end

%X = NormalizeFea(X,1);
X_all = X; % n x d
Y_all = Y(:);
% 设置十折交叉验证（现在受上面的随机种子影响，可重现）
cv = cvpartition(Y_all, 'KFold', 10); % 创建10折交叉验证分区
% 定义算法参数
params = struct();
params.lambda2 = 10;
params.K_keep = []; % 让算法自适应选择K
params.maxIter = 5;
params.tol = 1e-3;
params.k_density_values = [3, 7, 15]; % 多尺度密度估计
params.weights = []; % 自适应选择权重
params.lambda2_min = 0.1 * params.lambda2;
params.lambda2_max = 10 * params.lambda2;
% 设置算法随机种子
algorithm_seed = 3; % 1 or 2 or 3
rng(algorithm_seed, 'twister');

% 初始化存储KNN-Graph算法指标和时间性能的结构
metrics_folds = struct();
inference_times = struct();
training_times = struct(); % [新增] 存储训练时间

% 为KNN-Graph算法初始化空结构数组
% [修改] 预定义所有可能用到的字段，包括 train_time 和 obj_history
empty_struct = struct('metrics', [], 'fold_time', [], 'train_time', [], ...
                      'K_values', [], 'predicted_labels', [], ...
                      'lambda2_adaptive', [], 'sigma', [], 'obj_history', [], 'error', []);
metrics_folds.HNSW = repmat(empty_struct, cv.NumTestSets, 1);

% 进行十折交叉验证
for fold_i = 1:cv.NumTestSets
    % 获取当前折的训练和测试索引
    trainIdx = cv.training(fold_i);
    testIdx = cv.test(fold_i);
    
    % 划分当前折的数据
    Xtr = X_all(trainIdx, :);
    Ytr = Y_all(trainIdx);
    Xte = X_all(testIdx, :);
    Yte = Y_all(testIdx);
    
    % 运行KNN-Graph算法
    try
        % 设置当前折的参数
        params_current = params;
        params_current.Y = Ytr; % 训练标签
        
        % ==================== 开始记录训练时间 ====================
        train_tic = tic; 
        
        % 调用 lasso_self_rep_kernel 计算自适应K值和近邻，并获取目标函数历史 obj
        [Wtr, Ktr, sigma, lambda2_adaptive, adaptive_weights, obj] = lasso_self_rep_kernel(Xtr', params_current);
  
        % 提取 K_values 和 neighbor_indices
        ntr = size(Xtr, 1);
        K_values = zeros(ntr, 1);
        Neighbors = cell(ntr, 1);
        
        for idx = 1:ntr
            wj = Wtr(:, idx);
            [~, ord] = sort(abs(wj), 'descend');
            ord(ord == idx) = []; % 排除自身
            if ~isempty(params_current.K_keep)
                ord = ord(1:min(params_current.K_keep, numel(ord)));
            else
                ord = ord(abs(wj(ord)) > 0);
            end
            
            K_values(idx) = numel(ord);
            Neighbors{idx} = ord(:).';
        end
        
        % 将Neighbors转换为矩阵格式用于HNSW
        maxK = max(K_values);
        neighbor_indices = zeros(ntr, maxK);
        for idx = 1:ntr
            k = numel(Neighbors{idx});
            if k > 0
                neighbor_indices(idx, 1:k) = Neighbors{idx};
            end
        end
        
        % 调用 build_hnsw_index 构建HNSW索引和预计算标签
        [index, precomputed_labels] = build_hnsw_index(Xtr, Ytr, K_values, neighbor_indices);
        
        % ==================== 结束记录训练时间 ====================
        training_times(fold_i).hnsw = toc(train_tic);
        
        % 预测阶段
        test_size = size(Xte, 1);
        tic;
        if test_size > 500
            % 大规模测试集，分块处理
            num_blocks = ceil(test_size / 1000);
            predicted_labels = zeros(test_size, 1);
            
            for block = 1:num_blocks
                start_idx = (block-1)*1000 + 1;
                end_idx = min(block*1000, test_size);
                [block_pred, ~] = query_hnsw_index(index, Xte(start_idx:end_idx, :), precomputed_labels);
                predicted_labels(start_idx:end_idx) = block_pred;
            end
        else
            % 小规模测试集，直接处理
            [predicted_labels, ~] = query_hnsw_index(index, Xte, precomputed_labels);
        end
        inference_times(fold_i).hnsw = toc;
        
        % 评估性能并保存各项指标
        metrics = EvaluationMetrics(predicted_labels, Yte);
        metrics_folds.HNSW(fold_i).metrics = metrics;
        metrics_folds.HNSW(fold_i).fold_time = inference_times(fold_i).hnsw;
        metrics_folds.HNSW(fold_i).train_time = training_times(fold_i).hnsw; % [新增] 记录训练时间
        metrics_folds.HNSW(fold_i).K_values = K_values;
        metrics_folds.HNSW(fold_i).predicted_labels = predicted_labels;
        metrics_folds.HNSW(fold_i).lambda2_adaptive = lambda2_adaptive;
        metrics_folds.HNSW(fold_i).sigma = sigma;
        metrics_folds.HNSW(fold_i).obj_history = obj; % [新增] 记录当前折的目标函数收敛曲线
        metrics_folds.HNSW(fold_i).error = '';
        
    catch ME
        fprintf('KNN-Graph算法在第 %d 折运行出错: %s\n', fold_i, ME.message);
        % 记录错误信息
        metrics_folds.HNSW(fold_i).error = ME.message;
        metrics_folds.HNSW(fold_i).metrics = [];
        metrics_folds.HNSW(fold_i).fold_time = NaN;
        metrics_folds.HNSW(fold_i).train_time = NaN; % [新增] 错误时训练时间记为 NaN
        metrics_folds.HNSW(fold_i).obj_history = []; % [新增] 错误时置空
    end
end

% 计算KNN-Graph算法的平均指标
try
    avg_metrics = compute_avg_metrics(metrics_folds.HNSW);
catch ME
    fprintf('计算KNN-Graph平均指标时出错: %s\n', ME.message);
    avg_metrics = struct('accuracy', 0);
end

% 输出结果
if isfield(avg_metrics, 'accuracy') && ~isempty(avg_metrics.accuracy)
    accuracy = avg_metrics.accuracy * 100;
    fprintf('\n==========================================\n');
    fprintf('算法KNN-Graph的平均准确率: %.2f%%\n', accuracy);
    
    % 保存结果
    dataset_name = 'binalpha_uni';
    filename = sprintf('KNN-Graph_%s_algorithm_%.2f.mat', dataset_name, accuracy);
    
    % 计算KNN-Graph算法的平均推理时间和平均训练时间
    times_inf = [metrics_folds.HNSW.fold_time];
    avg_inference_time = mean(times_inf, 'omitnan');
    fold_times = times_inf;
    
    times_train = [metrics_folds.HNSW.train_time]; % [新增]
    avg_train_time = mean(times_train, 'omitnan'); % [新增]
    fold_train_times = times_train; % [新增]
    
    % [修改] 将平均训练时间和每折的训练时间加入保存列表
    save(filename, 'avg_metrics', 'metrics_folds', 'avg_inference_time', 'fold_times', 'avg_train_time', 'fold_train_times', 'algorithm_seed');
    fprintf('结果已保存到 %s\n', filename);
    
    % 输出详细指标
    fprintf('\n------------ 详细性能指标 ------------\n');
    fprintf('准确率: %.2f%%\n', avg_metrics.accuracy * 100);
    if isfield(avg_metrics, 'precision')
        fprintf('精确率: %.2f%%\n', avg_metrics.precision * 100);
    end
    if isfield(avg_metrics, 'recall')
        fprintf('召回率: %.2f%%\n', avg_metrics.recall * 100);
    end
    if isfield(avg_metrics, 'f1')
        fprintf('F1分数: %.2f%%\n', avg_metrics.f1 * 100);
    end
    
    fprintf('\n------------ 计算开销分析 ------------\n');
    fprintf('平均训练时间 (10折平均): %.4f 秒\n', avg_train_time);
    fprintf('平均推理时间 (10折平均): %.4f 秒\n', avg_inference_time);
    fprintf('==========================================\n');
else
    fprintf('算法KNN-Graph %d 无法计算准确率\n', algorithm_seed);
end

% compute_avg_metrics 函数
function avg_metrics = compute_avg_metrics(folds)
    % 检查输入是否有效
    if isempty(folds) || ~isfield(folds, 'metrics') || isempty([folds.metrics])
        avg_metrics = struct('accuracy', 0, 'precision', 0, 'recall', 0, 'f1', 0);
        return;
    end
    
    % 获取有效的指标（排除空值）
    valid_folds = folds(~cellfun(@isempty, {folds.metrics}));
    if isempty(valid_folds)
        avg_metrics = struct('accuracy', 0, 'precision', 0, 'recall', 0, 'f1', 0);
        return;
    end
    
    % 获取所有字段名
    field_names = fieldnames(valid_folds(1).metrics);
    avg_metrics = struct();
    
    for i = 1:length(field_names)
        field = field_names{i};
        values = [];
        for j = 1:length(valid_folds)
            if isfield(valid_folds(j).metrics, field)
                values(end+1) = valid_folds(j).metrics.(field);
            end
        end
        if ~isempty(values)
            avg_metrics.(field) = mean(values);
        else
            avg_metrics.(field) = 0;
        end
    end
end
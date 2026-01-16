function [index, precomputed_labels] = build_hnsw_index(X, Y, K_values, neighbor_indices)
% BUILD_HNSW_INDEX 基于HNSW构建索引，并计算加权MV预测标签
%
% 输入：
%   X - 训练数据，n x d 矩阵，n个样本，d维特征
%   Y - 训练数据的类标签，n x 1 向量
%   K_values - 每个训练点的自适应K值，n x 1 向量
%   neighbor_indices - 每个训练点的K近邻索引，n x max(K_values) 矩阵，未用到的位置填充0
%
% 输出：
%   index - HNSW索引对象，包含训练数据X的结构化表示
%   precomputed_labels - 每个训练点的加权MV预测标签，n x 1 向量

% 输入验证
[n, d] = size(X);
if ~isvector(Y) || length(Y) ~= n
    error('Y 必须是 n x 1 向量');
end
if ~isvector(K_values) || length(K_values) ~= n
    error('K_values 必须是 n x 1 向量');
end
if size(neighbor_indices, 1) ~= n
    error('neighbor_indices 必须是 n x max(K_values) 矩阵');
end

% 参数
eps = 1e-6; % 避免除零
num_classes = length(unique(Y)); % 类别数

% 步骤1: 计算每个训练点的加权MV预测标签
precomputed_labels = zeros(n, 1); % 存储预测标签
for i = 1:n
    % 获取当前点的K近邻索引和K值
    K = K_values(i);
    neigh_idx = neighbor_indices(i, 1:K); % 当前点的K近邻索引
    
    % 计算近邻到当前点的距离
    neigh_points = X(neigh_idx, :); % 近邻的特征
    curr_point = X(i, :); % 当前点
    neigh_dist = sqrt(sum((neigh_points - curr_point).^2, 2)); % 欧氏距离
    
    % 获取近邻的标签
    neigh_labels = Y(neigh_idx);
    
    % 加权多数投票
    weights = 1 ./ (neigh_dist + eps); % 近邻权重：距离倒数
    self_weight = 2 * max(weights); % 本节点权重：最近邻权重的2倍
    self_label = Y(i); % 本节点标签
    
    % 计算每个类别的加权得分
    class_scores = zeros(num_classes, 1);
    for c = 1:num_classes
        % 近邻贡献
        class_scores(c) = sum(weights(neigh_labels == c));
        % 本节点贡献
        if self_label == c
            class_scores(c) = class_scores(c) + self_weight;
        end
    end
    
    % 选择得分最高的类别
    [~, pred_class] = max(class_scores);
    precomputed_labels(i) = pred_class;
end

% 步骤2: 构建HNSW索引 - 修复参数设置
% 根据训练集大小动态调整参数
MaxNumLinksPerNode = min(32, n); % 确保不超过训练样本数
TrainSetSize = n; % 使用所有训练样本

% 检查参数有效性
if MaxNumLinksPerNode < 2
    MaxNumLinksPerNode = 2; % 最小值为2
end

if TrainSetSize < MaxNumLinksPerNode
    % 如果训练样本太少，调整参数
    MaxNumLinksPerNode = max(2, floor(n/2));
    TrainSetSize = n;
end

fprintf('构建HNSW索引参数: n=%d, MaxNumLinksPerNode=%d, TrainSetSize=%d\n', ...
    n, MaxNumLinksPerNode, TrainSetSize);

try
    index = hnswSearcher(X, 'Distance', 'euclidean', ...
                         'MaxNumLinksPerNode', MaxNumLinksPerNode, ...
                         'TrainSetSize', TrainSetSize);
catch ME
    fprintf('HNSW索引构建失败: %s\n', ME.message);
    fprintf('尝试使用备用参数...\n');
    
    % 备用方案：使用更保守的参数
    try
        MaxNumLinksPerNode = min(16, n);
        TrainSetSize = n;
        
        index = hnswSearcher(X, 'Distance', 'euclidean', ...
                             'MaxNumLinksPerNode', MaxNumLinksPerNode, ...
                             'TrainSetSize', TrainSetSize);
    catch ME2
        fprintf('备用方案也失败: %s\n', ME2.message);
        fprintf('使用ExhaustiveSearcher作为备选...\n');
        
        % 最终备选：使用穷举搜索
         index = ExhaustiveSearcher(X, 'Distance', 'euclidean');
    end
end

% 步骤3: 保存数据（供其他用途，可选）
save('hnsw_data.mat', 'X', 'Y', 'K_values', 'neighbor_indices', 'precomputed_labels');

end
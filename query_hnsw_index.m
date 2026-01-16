function [predicted_labels, query_time] = query_hnsw_index(index, X_test, precomputed_labels)
% QUERY_HNSW_INDEX 查询HNSW索引，返回测试点的预测标签和查询时间
%
% 输入：
%   index - HNSW索引对象（由build_hnsw_index生成）或ExhaustiveSearcher对象
%   X_test - 测试数据，num_test x d 矩阵
%   precomputed_labels - 训练点的加权MV预测标签，n x 1 向量
%
% 输出：
%   predicted_labels - 测试点的预测标签，num_test x 1 向量
%   query_time - 查询总时间（秒），包括knnsearch和标签提取
%
% 注意：
%   - 需要 Statistics and Machine Learning Toolbox (R2022b 或更高)
%   - 确保 X_test 的维度与训练数据 X 一致
%   - 确保 precomputed_labels 与 index 中的训练数据对应

% 验证输入
[n_test, d_test] = size(X_test);
if ~isobject(index)
    error('index 必须是有效的搜索器对象');
end

% 获取训练数据的特征维度
try
    % 尝试获取hnswSearcher的特征维度
    if isprop(index, 'X')
        d_train = size(index.X, 2);
        n_train = size(index.X, 1);
    elseif isprop(index, 'NumFeatures')
        % 某些版本的hnswSearcher可能有NumFeatures属性
        d_train = index.NumFeatures;
        n_train = index.NumObservations;
    else
        % 如果无法直接获取，使用测试数据维度
        d_train = d_test;
        n_train = length(precomputed_labels);
    end
catch
    % 如果所有方法都失败，使用测试数据维度
    d_train = d_test;
    n_train = length(precomputed_labels);
end

% 检查维度一致性
if d_test ~= d_train
    error('X_test 的维度 (%d) 必须与训练数据维度 (%d) 一致', d_test, d_train);
end

% 检查precomputed_labels长度
if ~isvector(precomputed_labels) || length(precomputed_labels) ~= n_train
    error('precomputed_labels 长度 (%d) 必须与训练数据点数 (%d) 一致', ...
          length(precomputed_labels), n_train);
end

% 开始计时
tic;

% 查询每个测试点的最近邻（1-NN）
[nearest_idx, ~] = knnsearch(index, X_test, 'K', 1);

% 获取最近邻的预计算标签
predicted_labels = precomputed_labels(nearest_idx);

% 结束计时
query_time = toc;

% 输出查询统计信息
fprintf('查询完成: %d个测试样本，耗时 %.4f秒\n', n_test, query_time);

end
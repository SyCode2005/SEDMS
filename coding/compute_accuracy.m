function acc = compute_accuracy(ground_lables, actual_ids, k)

% 检查输入长度是否一致
if length(ground_lables) ~= length(actual_ids)
    error('ground_lables 和 actual_ids 的长度不一致');
end

% 检查 actual_ids 的值范围
if min(actual_ids) < 1 || max(actual_ids) > k
    error('actual_ids 的值超出了 1 到 k 的范围');
end

total_cluster_num = 0;
for idx = 1 : k
    table = tabulate(ground_lables(actual_ids == idx));
    if ~isempty(table)
        [~, row_idx] = max(table(:,3));
        total_cluster_num = total_cluster_num + table(row_idx, 2);
    end
end
acc = total_cluster_num / length(ground_lables);
end


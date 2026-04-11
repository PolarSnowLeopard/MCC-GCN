# 把实验数据中的部分数据加到FT数据中，看一下模型性能
# 在运行相关脚本构建好HKU_data_6_FT_minoxidil_balanced.csv和HKU_data_6_experiment.csv后，运行该脚本
# 该脚本会直接从实验数据各类别中随机抽取相同数量的数据，并加到HKU_data_6_FT_minoxidil_balanced.csv中
# 以确保新的HKU_data_6_FT_minoxidil_balanced.csv依然是平衡数据集
# 同时也会更新HKU_data_6_experiment{, _1, _2}.csv，以确保预测数据集中不包含加入微调阶段的实验数据
import pandas as pd

sample_num = {
    'failed': 3,
    'salt': 5,
    'cocrystal': 3,
    'hydrate': 3
}

# 读取实验数据
exp_data_1 = pd.read_csv('HKU_data_6_experiment_1.csv')

# 从实验数据每个类别中随机抽取2个数据,然后删去exp_data中所有选中数据
exp_data_1['temp_id'] = range(len(exp_data_1))
selected_exp_data_1 = exp_data_1.groupby('label_str').apply(
    # lambda x: x.sample(n=min(len(x)-2, 9))
    # lambda x: x.sample(n=int(len(x) * 0.6))
    lambda x: x.sample(n=0)
    # lambda x: x.sample(n=sample_num[x['label_str'].values[0]])
).reset_index(drop=True)
selected_ids = selected_exp_data_1['temp_id'].values
exp_data_1 = exp_data_1[~exp_data_1['temp_id'].isin(selected_ids)]
exp_data_1 = exp_data_1.drop('temp_id', axis=1)
selected_exp_data_1 = selected_exp_data_1.drop('temp_id', axis=1)

# 构造逆序后的新的exp_data_2
exp_data_2 = exp_data_1.copy()
exp_data_2[['reactant_A', 'reactant_B']] = exp_data_2[['reactant_B', 'reactant_A']]

# 保存三个实验数据集
exp_data_1.to_csv('HKU_data_6_experiment_1.csv', header=True, index=False)
exp_data_2.to_csv('HKU_data_6_experiment_2.csv', header=True, index=False)
exp_data = pd.concat([exp_data_1, exp_data_2], ignore_index=True)
exp_data.to_csv('HKU_data_6_experiment.csv', header=True, index=False)


print(len(exp_data_1))
print(len(exp_data_2))
print(len(exp_data))

# 将反应物A和B互换，进行数据增强
selected_exp_data_2 = selected_exp_data_1.copy()
selected_exp_data_2[['reactant_A', 'reactant_B']] = selected_exp_data_2[['reactant_B', 'reactant_A']]

# 读取FT数据
FT_data = pd.read_csv('HKU_data_6_FT_minoxidil_balanced.csv')
# FT_data = pd.read_csv('HKU_data_6_FT_minoxidil.csv')

# 将实验数据和FT数据合并
combined_data = pd.concat([FT_data, selected_exp_data_1, selected_exp_data_2], ignore_index=True)

# 保存合并后的数据
# combined_data.to_csv('HKU_data_6_FT_minoxidil_balanced_with_exp.csv', header=True, index=False)
# combined_data.to_csv('HKU_data_6_FT_minoxidil_with_exp.csv', header=True, index=False)

print(combined_data.value_counts('label_str'))
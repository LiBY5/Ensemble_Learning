# 基础环境准备
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# 设置绘图风格，使图表更美观
sns.set_style("whitegrid")

# 加载数据
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
submission_sample = pd.read_csv('sample_submission.csv')

# 立即检查数据是否加载成功
print("训练集形状：", train.shape)
print("测试集形状：", test.shape)
print("提交示例形状：", submission_sample.shape)
print("\n训练集前几行：")
print(train.head())
print("\n训练集基本信息：")
train.info()

# 查看目标变量的基本统计信息
print(train['SalePrice'].describe())
# 绘制分布直方图
plt.figure(figsize=(10, 6))
sns.histplot(train['SalePrice'], kde=True, bins=30)
plt.title('SalePrice Distribution')
plt.savefig('SalePrice Distribution.png')
plt.show()

# 计算缺失值比例并按降序排列
missing_ratio = (train.isnull().sum() / len(train) * 100).sort_values(ascending=False)
missing_ratio = missing_ratio[missing_ratio > 0]  # 只显示有缺失的特征
print("训练集中有缺失的特征及其比例：")
print(missing_ratio.head(20))  # 先看缺失最严重的20个

# --- 模块2.1：目标变量分析 ---
print("="*50)
print("模块2.1：目标变量分析")
print("="*50)

# 1. 绘制SalePrice的分布与Q-Q图，检验正态性
from scipy import stats
from scipy.stats import norm

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# 原始分布
sns.histplot(train['SalePrice'], kde=True, ax=axes[0], bins=30)
axes[0].axvline(train['SalePrice'].mean(), color='r', linestyle='--', label=f'Mean: {train["SalePrice"].mean():.0f}')
axes[0].axvline(train['SalePrice'].median(), color='g', linestyle='--', label=f'Median: {train["SalePrice"].median():.0f}')
axes[0].set_title('Original SalePrice Distribution')
axes[0].legend()
axes[0].set_xlabel('SalePrice')

# 对数变换后的分布
train['SalePrice_log'] = np.log1p(train['SalePrice']) # 使用log1p防止log(0)的情况
sns.histplot(train['SalePrice_log'], kde=True, ax=axes[1], bins=30)
axes[1].axvline(train['SalePrice_log'].mean(), color='r', linestyle='--', label=f'Mean: {train["SalePrice_log"].mean():.2f}')
axes[1].axvline(train['SalePrice_log'].median(), color='g', linestyle='--', label=f'Median: {train["SalePrice_log"].median():.2f}')
axes[1].set_title('Log-Transformed SalePrice Distribution')
axes[1].legend()
axes[1].set_xlabel('Log(SalePrice)')

# Q-Q图 (针对变换后的数据)
stats.probplot(train['SalePrice_log'], dist="norm", plot=axes[2])
axes[2].set_title('Q-Q Plot for Log(SalePrice)')

plt.tight_layout()
plt.show()

# 输出偏度和峰度
print(f"原始SalePrice偏度(Skewness): {train['SalePrice'].skew():.4f}")
print(f"对数变换后偏度: {train['SalePrice_log'].skew():.4f}")
print(f"原始SalePrice峰度(Kurtosis): {train['SalePrice'].kurt():.4f}")
print(f"对数变换后峰度: {train['SalePrice_log'].kurt():.4f}")

# 2. 异常值检测 - 通过散点图观察与关键连续变量的关系
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
key_numeric_features = ['GrLivArea', 'TotalBsmtSF', '1stFlrSF', 'LotArea', 'OverallQual', 'YearBuilt']
for idx, feature in enumerate(key_numeric_features):
    row, col = idx // 3, idx % 3
    sns.scatterplot(data=train, x=feature, y='SalePrice', ax=axes[row, col], alpha=0.6)
    axes[row, col].set_title(f'SalePrice vs {feature}')
plt.tight_layout()
plt.show()

# 3. 与所有数值特征的相关性分析 (初步)
print("\n与SalePrice相关性最高的10个特征（数值型）：")
numeric_features = train.select_dtypes(include=[np.number]).columns
correlation_with_price = train[numeric_features].corr()['SalePrice'].sort_values(ascending=False)
print(correlation_with_price.head(11)) # 包括SalePrice自己

# --- 模块2.2 & 2.3：数据合并、异常值处理与特征工程准备 ---
print("=" * 50)
print("模块2.2 & 2.3：数据预处理与特征工程")
print("=" * 50)

# 1. 合并训练集和测试集，以便统一进行特征工程
train['is_train'] = 1
test['is_train'] = 0
# 测试集没有SalePrice，我们添加一列以方便合并
test['SalePrice'] = np.nan
# 同时保存训练集的目标变量（对数变换后）
y_train_log = train['SalePrice_log'].copy()

data = pd.concat([train, test], ignore_index=True, sort=False)
print(f"合并后数据形状: {data.shape}")

# 2. 删除GrLivArea的异常值 (仅在训练集中标识并删除)
# 注意：此操作仅针对训练集，我们最终会从data中分离出干净的train_data
outlier_idx = data[(data['GrLivArea'] > 4000) & (data['SalePrice'] < 300000) & (data['is_train'] == 1)].index
print(f"将删除的训练集异常值索引: {list(outlier_idx)}")
data_clean = data.drop(outlier_idx, axis=0).reset_index(drop=True)
# 同时更新y_train_log
y_train_log_clean = y_train_log.drop(outlier_idx, axis=0).reset_index(drop=True)
print(f"删除异常值后数据形状: {data_clean.shape}")

# 3. 深入检查缺失值（结合数据字典）
# 我们根据数据字典和领域知识，将缺失分为几类处理
# 首先，列出所有有缺失的特征
missing_features = data_clean.isnull().sum()
missing_features = missing_features[missing_features > 0]
print(f"\n合并数据集中所有有缺失的特征（共{len(missing_features)}个）:")
print(missing_features.sort_values(ascending=False))


# 定义处理函数
def handle_missing_values(df):
    """
    根据数据字典和业务逻辑，系统性处理缺失值。
    策略：
    - 有意义的缺失（None）：用 ‘None‘, ‘NA‘, 或 0 填充。
    - 真正的缺失：用众数、中位数或均值填充。
    """
    df_processed = df.copy()

    # 根据数据字典，这些分类特征的NaN代表“无”
    none_cat_features = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu',
                         'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
                         'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
                         'MasVnrType', 'MSSubClass']  # MSSubClass的‘None’是特殊的
    for col in none_cat_features:
        if col in df_processed.columns:
            df_processed[col] = df_processed[col].fillna('None')

    # 这些数值特征的NaN也代表“无”，用0填充
    none_num_features = ['MasVnrArea', 'GarageYrBlt', 'GarageArea', 'GarageCars',
                         'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF',
                         'BsmtFullBath', 'BsmtHalfBath', 'PoolArea']
    for col in none_num_features:
        if col in df_processed.columns:
            df_processed[col] = df_processed[col].fillna(0)

    # 其他缺失处理（真正的缺失）
    # LotFrontage（临街距离）：通常用同社区的中位数填充
    df_processed['LotFrontage'] = df_processed.groupby('Neighborhood')['LotFrontage'].transform(
        lambda x: x.fillna(x.median())
    )
    # 如果还有缺失（比如某个社区所有样本都缺失），用全局中位数填充
    df_processed['LotFrontage'].fillna(df_processed['LotFrontage'].median(), inplace=True)

    # 只有一个缺失值的特征，用众数填充
    df_processed['Electrical'] = df_processed['Electrical'].fillna(df_processed['Electrical'].mode()[0])
    df_processed['KitchenQual'] = df_processed['KitchenQual'].fillna(df_processed['KitchenQual'].mode()[0])  # 测试集可能有缺失
    df_processed['Exterior1st'] = df_processed['Exterior1st'].fillna(df_processed['Exterior1st'].mode()[0])
    df_processed['Exterior2nd'] = df_processed['Exterior2nd'].fillna(df_processed['Exterior2nd'].mode()[0])
    df_processed['SaleType'] = df_processed['SaleType'].fillna(df_processed['SaleType'].mode()[0])
    df_processed['MSZoning'] = df_processed['MSZoning'].fillna(df_processed['MSZoning'].mode()[0])
    df_processed['Functional'] = df_processed['Functional'].fillna(df_processed['Functional'].mode()[0])

    # 检查是否还有缺失
    remaining_missing = df_processed.isnull().sum().sum()
    print(f"缺失值处理完成。剩余缺失值总数: {remaining_missing}")
    if remaining_missing > 0:
        print("剩余缺失的特征：")
        print(df_processed.isnull().sum()[df_processed.isnull().sum() > 0])

    return df_processed


# 4. 应用缺失值处理
print("\n开始处理缺失值...")
data_processed = handle_missing_values(data_clean)

# 5. 特征构造（您的计划中的模块2.3策略）
print("\n开始构造新特征...")
# 总面积 = 地上面积 + 地下室面积
data_processed['TotalSF'] = data_processed['TotalBsmtSF'] + data_processed['1stFlrSF'] + data_processed['2ndFlrSF']
data_processed['TotalArea'] = data_processed['TotalBsmtSF'] + data_processed['GrLivArea']

# 房屋年龄相关
data_processed['HouseAge'] = data_processed['YrSold'] - data_processed['YearBuilt']
data_processed['RemodAge'] = data_processed['YrSold'] - data_processed['YearRemodAdd']
# 避免负年龄（可能是数据错误），用0替代
data_processed['HouseAge'] = data_processed['HouseAge'].apply(lambda x: x if x >= 0 else 0)
data_processed['RemodAge'] = data_processed['RemodAge'].apply(lambda x: x if x >= 0 else 0)

# 浴室总数
data_processed['TotalBath'] = data_processed['FullBath'] + 0.5 * data_processed['HalfBath'] + \
                              data_processed['BsmtFullBath'] + 0.5 * data_processed['BsmtHalfBath']
# 房间面积比
data_processed['GrLivArea_per_Room'] = data_processed['GrLivArea'] / data_processed['TotRmsAbvGrd']
data_processed['GrLivArea_per_Room'].replace([np.inf, -np.inf], 0, inplace=True)  # 处理除零错误

# 是否有地下室、车库、壁炉、游泳池等（二值特征）
data_processed['HasBasement'] = (data_processed['TotalBsmtSF'] > 0).astype(int)
data_processed['HasGarage'] = (data_processed['GarageArea'] > 0).astype(int)
data_processed['HasFireplace'] = (data_processed['Fireplaces'] > 0).astype(int)
data_processed['HasPool'] = (data_processed['PoolArea'] > 0).astype(int)
data_processed['Has2ndFloor'] = (data_processed['2ndFlrSF'] > 0).astype(int)

print(
    f"特征构造完成。新增了 {len(['TotalSF', 'TotalArea', 'HouseAge', 'RemodAge', 'TotalBath', 'GrLivArea_per_Room', 'HasBasement', 'HasGarage', 'HasFireplace', 'HasPool', 'Has2ndFloor'])} 个新特征。")

# 6. 重新分离训练集和测试集
train_processed = data_processed[data_processed['is_train'] == 1].copy()
test_processed = data_processed[data_processed['is_train'] == 0].copy()

# 从训练集中移除我们添加的辅助列，并恢复目标变量
# 注意：我们已经用 y_train_log_clean 保存了清洗并变换后的目标变量
train_final = train_processed.drop(['SalePrice', 'is_train'], axis=1)
# 但为了后续分析，我们保留一列原始SalePrice
train_final['SalePrice_log'] = y_train_log_clean.values

test_final = test_processed.drop(['SalePrice', 'SalePrice_log', 'is_train'], axis=1)

print(f"\n预处理完成！")
print(f"最终训练集形状: {train_final.shape}")
print(f"最终测试集形状: {test_final.shape}")
print(f"训练集目标变量 (SalePrice_log) 形状: {train_final['SalePrice_log'].shape}")

# --- 修复缺失值并完成特征编码 ---
print("="*50)
print("模块2.3（续）：修复缺失值并完成特征编码")
print("="*50)

# 1. 修复 Utilities 特征的缺失值（在合并的数据 data_processed 上操作）
data_processed['Utilities'] = data_processed['Utilities'].fillna(data_processed['Utilities'].mode()[0])
print(f"修复 Utilities 缺失值后，总缺失数: {data_processed.isnull().sum().sum()} (应仅为测试集的SalePrice和SalePrice_log)")

# 2. 特征编码准备
# 重新分离训练集和测试集（修复缺失后）
train_processed = data_processed[data_processed['is_train'] == 1].copy()
test_processed = data_processed[data_processed['is_train'] == 0].copy()

# 准备目标变量
y_train = train_processed['SalePrice_log'].copy()  # 对数变换后的目标
train_ids = train_processed['Id'].copy()
test_ids = test_processed['Id'].copy()

# 移除不需要的列
features_to_drop = ['Id', 'SalePrice', 'SalePrice_log', 'is_train']
X_train = train_processed.drop(features_to_drop, axis=1)
X_test = test_processed.drop(features_to_drop, axis=1)

print(f"\n编码前特征形状:")
print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"X_test: {X_test.shape}")

# 3. 区分特征类型
numeric_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = X_train.select_dtypes(include=['object']).columns.tolist()
print(f"\n数值特征数量: {len(numeric_features)}")
print(f"分类特征数量: {len(categorical_features)}")

# 4. 分类特征编码
from sklearn.preprocessing import LabelEncoder

# 4.1 处理有序分类特征（质量、条件等，通常有内在顺序）
# 根据数据字典定义顺序映射
ordinal_mapping = {
    'ExterQual': {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
    'ExterCond': {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
    'BsmtQual': {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
    'BsmtCond': {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
    'HeatingQC': {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
    'KitchenQual': {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
    'FireplaceQu': {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
    'GarageQual': {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
    'GarageCond': {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
    'PoolQC': {'None': 0, 'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4},
    'BsmtExposure': {'None': 0, 'No': 1, 'Mn': 2, 'Av': 3, 'Gd': 4},
    'BsmtFinType1': {'None': 0, 'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ': 6},
    'BsmtFinType2': {'None': 0, 'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ': 6},
    'GarageFinish': {'None': 0, 'Unf': 1, 'RFn': 2, 'Fin': 3},
    'Fence': {'None': 0, 'MnWw': 1, 'GdWo': 2, 'MnPrv': 3, 'GdPrv': 4},
    'LotShape': {'IR3': 1, 'IR2': 2, 'IR1': 3, 'Reg': 4}, # 假设Reg最规整
    'LandSlope': {'Sev': 1, 'Mod': 2, 'Gtl': 3}, # 假设Gtl坡度最缓
    'PavedDrive': {'N': 0, 'P': 1, 'Y': 2}
}

# 应用有序编码
for col, mapping in ordinal_mapping.items():
    if col in X_train.columns:
        X_train[col] = X_train[col].map(mapping)
        X_test[col] = X_test[col].map(mapping)
        # 将编码后的列转为数值类型
        X_train[col] = pd.to_numeric(X_train[col], errors='coerce').fillna(0).astype(int)
        X_test[col] = pd.to_numeric(X_test[col], errors='coerce').fillna(0).astype(int)
        # 将此列从分类特征列表移到数值特征列表
        if col in categorical_features:
            categorical_features.remove(col)
            numeric_features.append(col)

# 4.2 处理名义分类特征（无内在顺序）
# 对于基数较低（唯一值较少）的特征，使用独热编码
# 对于基数非常高（如超过20个唯一值）的特征，使用目标编码（更稳健，避免维度爆炸）
from sklearn.preprocessing import OneHotEncoder
from category_encoders import TargetEncoder
import warnings
warnings.filterwarnings('ignore')

# 识别高基数特征
high_cardinality_features = [col for col in categorical_features if X_train[col].nunique() > 20]
low_cardinality_features = [col for col in categorical_features if X_train[col].nunique() <= 20]

print(f"\n高基数分类特征 (>20个唯一值): {high_cardinality_features}")
print(f"低基数分类特征 (<=20个唯一值): {len(low_cardinality_features)} 个")

# 目标编码 (在高基数特征上，使用训练集拟合，避免数据泄露)
if high_cardinality_features:
    encoder_high = TargetEncoder(cols=high_cardinality_features)
    X_train[high_cardinality_features] = encoder_high.fit_transform(X_train[high_cardinality_features], y_train)
    X_test[high_cardinality_features] = encoder_high.transform(X_test[high_cardinality_features])

# 独热编码 (在低基数特征上)
if low_cardinality_features:
    # 确保训练集和测试集编码一致
    encoder_low = OneHotEncoder(sparse_output=False, handle_unknown='ignore', dtype=np.int32)
    # 拟合训练集
    encoded_train = encoder_low.fit_transform(X_train[low_cardinality_features])
    encoded_test = encoder_low.transform(X_test[low_cardinality_features])
    # 创建新列名
    new_col_names = encoder_low.get_feature_names_out(low_cardinality_features)
    # 转换为DataFrame并拼接
    encoded_train_df = pd.DataFrame(encoded_train, columns=new_col_names, index=X_train.index)
    encoded_test_df = pd.DataFrame(encoded_test, columns=new_col_names, index=X_test.index)
    # 移除原始分类列，添加编码后的列
    X_train = X_train.drop(low_cardinality_features, axis=1)
    X_test = X_test.drop(low_cardinality_features, axis=1)
    X_train = pd.concat([X_train, encoded_train_df], axis=1)
    X_test = pd.concat([X_test, encoded_test_df], axis=1)

# 5. 最终检查
print(f"\n编码完成！")
print(f"最终特征矩阵形状 - X_train: {X_train.shape}")
print(f"最终特征矩阵形状 - X_test: {X_test.shape}")
print(f"目标变量形状 - y_train: {y_train.shape}")
print(f"特征类型: 全部为数值型，共 {X_train.shape[1]} 个特征。")

# 确保训练集和测试集列顺序完全一致
X_test = X_test.reindex(columns=X_train.columns, fill_value=0) # 对于测试集出现训练集未见过的类别，用0填充

# 保存处理好的数据，以备下一阶段使用
X_train.to_csv('X_train_processed.csv', index=False)
X_test.to_csv('X_test_processed.csv', index=False)
y_train.to_csv('y_train_log.csv', index=False, header=['SalePrice_log'])
pd.DataFrame({'Id': train_ids}).to_csv('train_ids.csv', index=False)
pd.DataFrame({'Id': test_ids}).to_csv('test_ids.csv', index=False)
print("\n预处理及编码后的特征和目标变量已保存为CSV文件。")
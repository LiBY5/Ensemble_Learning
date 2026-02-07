# 阶段三：基准模型建立
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error

# 加载预处理后的数据
X_train = pd.read_csv('X_train_processed.csv')
y_train = pd.read_csv('y_train_log.csv').values.ravel()  # 转换为1D数组
X_test = pd.read_csv('X_test_processed.csv')
print("数据加载完成。")
print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}")


# 任务3.1：建立评估框架
# 由于我们已对目标变量进行对数变换，使用RMSE等价于RMSLE
def rmse_cv(model, X, y, cv=5):
    """
    计算模型在交叉验证下的RMSE（负的，因为cross_val_score默认取负）
    """
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    rmse = np.sqrt(-cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv=kf, n_jobs=-1))
    return rmse


# 任务3.2：构建基准模型
# 我们尝试多种模型，并比较它们的交叉验证RMSE
models = {
    'Ridge': Ridge(alpha=1.0, random_state=42),
    'Lasso': Lasso(alpha=0.0005, random_state=42, max_iter=10000),
    'ElasticNet': ElasticNet(alpha=0.0005, l1_ratio=0.9, random_state=42, max_iter=10000),
    'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    'XGBoost': XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1, verbosity=0),
    'LightGBM': LGBMRegressor(n_estimators=100, random_state=42, n_jobs=-1, verbose=-1),
    'SVR': SVR(kernel='linear', C=1.0),
    'KNN': KNeighborsRegressor(n_neighbors=5, n_jobs=-1)
}

# 对于线性模型和SVR、KNN，我们需要对特征进行标准化
# 我们将数据标准化，注意：用训练集的均值和方差来标准化训练集和测试集
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 评估每个模型
results = {}
for name, model in models.items():
    if name in ['Ridge', 'Lasso', 'ElasticNet', 'SVR', 'KNN']:
        X = X_train_scaled
    else:
        X = X_train.values  # 树模型不需要标准化
    scores = rmse_cv(model, X, y_train, cv=5)
    results[name] = (scores.mean(), scores.std())
    print(f"{name}: 平均RMSE = {scores.mean():.5f} (±{scores.std():.5f})")

# 将结果转换为DataFrame便于比较
results_df = pd.DataFrame(results, index=['RMSE均值', 'RMSE标准差']).T
results_df = results_df.sort_values(by='RMSE均值')
print("\n模型性能排序：")
print(results_df)

# 可视化模型比较
import matplotlib.pyplot as plt

#设置字体为系统自带的中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 设置中文字体
# 解决负号显示问题
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方块的问题

plt.figure(figsize=(10, 6))
plt.barh(range(len(results_df)), results_df['RMSE均值'], xerr=results_df['RMSE标准差'], align='center', alpha=0.8)
plt.yticks(range(len(results_df)), results_df.index)
plt.xlabel('交叉验证RMSE (越低越好)')
plt.title('基准模型性能比较')
plt.gca().invert_yaxis()  # 倒置Y轴，使最好的模型在顶部
plt.tight_layout()
plt.show()

# 选择表现最好的几个模型，进行简单平均集成
print("\n--- 尝试简单平均集成 ---")
# 选择RMSE最低的3个模型
top_models = results_df.head(3).index.tolist()
print(f"选择前3个模型进行平均: {top_models}")

# 为简单平均集成准备预测结果
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin, clone


class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, models):
        self.models = models

    def fit(self, X, y):
        # 克隆模型以避免修改原始模型
        self.models_ = [clone(x) for x in self.models]
        # 训练每个模型
        for model in self.models_:
            model.fit(X, y)
        return self

    def predict(self, X):
        predictions = np.column_stack([model.predict(X) for model in self.models_])
        return np.mean(predictions, axis=1)


# 根据模型类型选择是否使用标准化数据
models_dict = {
    'Ridge': Ridge(alpha=1.0, random_state=42),
    'Lasso': Lasso(alpha=0.0005, random_state=42, max_iter=10000),
    'ElasticNet': ElasticNet(alpha=0.0005, l1_ratio=0.9, random_state=42, max_iter=10000),
    'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    'XGBoost': XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1, verbosity=0),
    'LightGBM': LGBMRegressor(n_estimators=100, random_state=42, n_jobs=-1, verbose=-1),
    'SVR': SVR(kernel='linear', C=1.0),
    'KNN': KNeighborsRegressor(n_neighbors=5, n_jobs=-1)
}

# 选择前3个模型
selected_models = [models_dict[name] for name in top_models]
averaged_models = AveragingModels(selected_models)

# 评估简单平均模型
# 注意：我们需要为每个模型准备正确的数据（是否标准化）
# 为了简化，我们这里使用原始数据（树模型）或标准化数据（线性模型）的混合？
# 更严谨的做法是为每个模型分别准备数据，但这里我们假设所有模型都使用标准化数据（或都不标准化）进行近似评估。
# 由于我们的top_models可能包含不同类型，我们统一使用标准化数据评估（因为线性模型需要标准化，而树模型对标准化不敏感）。
scores_avg = rmse_cv(averaged_models, X_train_scaled, y_train, cv=5)
print(f"简单平均模型({top_models}): 平均RMSE = {scores_avg.mean():.5f} (±{scores_avg.std():.5f})")

# 将简单平均模型的结果添加到比较中
results_df.loc['Averaging'] = (scores_avg.mean(), scores_avg.std())
results_df = results_df.sort_values(by='RMSE均值')
print("\n加入简单平均集成后的模型性能排序：")
print(results_df)
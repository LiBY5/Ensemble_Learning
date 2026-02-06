import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

# 1. 加载数据
california = fetch_california_housing()
X, y = california.data, california.target
feature_names = california.feature_names

# 2. 快速查看数据形态
print(f"特征数据X的形状: {X.shape}")
print(f"目标值y的形状: {y.shape}")
print(f"特征名称: {feature_names}")
print(f"前5个样本的目标值: {y[:5]}")

# 3. 划分训练集和测试集 (保持80%训练，20%测试)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\n训练集大小: {X_train.shape}")
print(f"测试集大小: {X_test.shape}")

from sklearn.ensemble import StackingRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score

# 1. 定义基模型列表
base_models = [
    ('dt', DecisionTreeRegressor(max_depth=5, random_state=42)),  # 限制深度防止过拟合
    ('rf', RandomForestRegressor(n_estimators=50, random_state=42)), # 控制树的数量以加速
    ('gbr', GradientBoostingRegressor(n_estimators=50, random_state=42)),
    ('svr', SVR(kernel='linear', C=1.0))  # 使用线性核，RBF核可能较慢
]

# 2. 定义元模型
meta_model = Ridge(alpha=1.0)

# 3. 创建Stacking回归器
# cv=5 表示使用5折交叉验证生成元特征
stacking_model = StackingRegressor(
    estimators=base_models,
    final_estimator=meta_model,
    cv=5,
    n_jobs=-1  # 使用所有CPU核心并行计算
)

# 4. 训练Stacking模型 (这会花费一点时间，因为要训练 cv * len(base_models) 个模型)
print("开始训练Stacking模型...")
stacking_model.fit(X_train, y_train)
print("Stacking模型训练完成！")

# 5. 在训练集和测试集上进行预测
y_train_pred = stacking_model.predict(X_train)
y_test_pred = stacking_model.predict(X_test)

# 6. 评估性能
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

print(f"\nStacking模型性能:")
print(f"  训练集 RMSE: {train_rmse:.4f}, R²: {train_r2:.4f}")
print(f"  测试集 RMSE: {test_rmse:.4f}, R²: {test_r2:.4f}")

print("4"*60)
# 初始化一个字典来存储结果
model_results = {}

# 遍历每个基模型，单独训练和评估
for model_name, model in base_models:
    # 复制模型，因为我们要重新训练
    m = model
    m.fit(X_train, y_train)
    y_pred = m.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    model_results[model_name] = {'RMSE': rmse, 'R2': r2}
    print(f"{model_name:>5} - 测试集 RMSE: {rmse:.4f}, R²: {r2:.4f}")

# 将Stacking模型的结果也加入字典
model_results['stacking'] = {'RMSE': test_rmse, 'R2': test_r2}
print(f"\nstacking - 测试集 RMSE: {test_rmse:.4f}, R²: {test_r2:.4f}")

print("5"*60)
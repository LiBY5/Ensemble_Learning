# 阶段四：Stacking集成实现
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error
import warnings

warnings.filterwarnings('ignore')

# 加载数据
X_train = pd.read_csv('X_train_processed.csv').values
y_train = pd.read_csv('y_train_log.csv').values.ravel()
X_test = pd.read_csv('X_test_processed.csv').values
print("数据加载完成，准备进行Stacking集成。")

# 模块4.1：第一层模型选择与训练（生成元特征）
print("\n" + "=" * 60)
print("模块4.1：第一层模型训练与元特征生成")
print("=" * 60)

# 根据阶段三结果，选择5个表现良好且异构的模型作为基模型
base_models = {
    'lasso': Lasso(alpha=0.0005, random_state=42, max_iter=20000),
    'ridge': Ridge(alpha=10.0, random_state=42),  # Ridge参数稍作调整以增加多样性
    'lightgbm': LGBMRegressor(n_estimators=100, learning_rate=0.05, random_state=42, verbose=-1),
    'random_forest': RandomForestRegressor(n_estimators=100, max_features=0.3, random_state=42, n_jobs=-1),
    'svr': SVR(kernel='rbf', C=20, epsilon=0.01)  # 使用RBF核的SVR
}

# 设置交叉验证
n_folds = 5
kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

# 初始化存储元特征的数组
train_meta_features = np.zeros((X_train.shape[0], len(base_models)))
test_meta_features = np.zeros((X_test.shape[0], len(base_models)))
test_meta_features_folds = np.zeros((X_test.shape[0], len(base_models), n_folds))

print(f"使用 {n_folds} 折交叉验证生成元特征...")

for i, (name, model) in enumerate(base_models.items()):
    print(f"  处理基模型: {name:15}", end="")
    test_fold_preds = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y_train)):
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]

        # 训练基模型
        model.fit(X_tr, y_tr)
        # 生成验证集预测（元特征）
        val_pred = model.predict(X_val)
        train_meta_features[val_idx, i] = val_pred
        # 生成测试集预测（每个折的模型都会对测试集预测一次）
        test_pred = model.predict(X_test)
        test_meta_features_folds[:, i, fold] = test_pred

    # 对测试集的预测，我们取各折预测的平均值，以降低方差
    test_meta_features[:, i] = test_meta_features_folds[:, i, :].mean(axis=1)
    # 计算该基模型在全体训练集上的性能（基于OOF预测）
    model_rmse = np.sqrt(mean_squared_error(y_train, train_meta_features[:, i]))
    print(f"  OOF RMSE: {model_rmse:.5f}")

print("第一层元特征生成完成。")
print(f"训练集元特征形状: {train_meta_features.shape}")
print(f"测试集元特征形状: {test_meta_features.shape}")

# 模块4.2：元模型设计与训练
print("\n" + "=" * 60)
print("模块4.2：元模型训练与比较")
print("=" * 60)

# 我们尝试三种不同的元模型
meta_models = {
    'Ridge_Meta': Ridge(alpha=1.0, random_state=42),
    'LightGBM_Meta': LGBMRegressor(n_estimators=150, learning_rate=0.05, random_state=42, verbose=-1),
    'Linear_Meta': Ridge(alpha=0.1, random_state=42)  # 更简单的线性模型
}

# 评估不同元模型在元特征上的性能
meta_results = {}
for name, model in meta_models.items():
    # 在元特征上使用交叉验证
    scores = []
    for train_idx, val_idx in kf.split(train_meta_features, y_train):
        X_tr_meta, X_val_meta = train_meta_features[train_idx], train_meta_features[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]

        model.fit(X_tr_meta, y_tr)
        val_pred = model.predict(X_val_meta)
        score = np.sqrt(mean_squared_error(y_val, val_pred))
        scores.append(score)

    mean_rmse, std_rmse = np.mean(scores), np.std(scores)
    meta_results[name] = (mean_rmse, std_rmse)
    print(f"{name:15} 平均RMSE = {mean_rmse:.5f} (±{std_rmse:.5f})")

# 选择最佳元模型
best_meta_name = min(meta_results, key=lambda x: meta_results[x][0])
best_meta_model = meta_models[best_meta_name]
print(f"\n选择最佳元模型: {best_meta_name}")

# 使用全部元特征重新训练最佳元模型
best_meta_model.fit(train_meta_features, y_train)

# 生成最终Stacking预测 (仍在对数空间)
stacking_train_pred = best_meta_model.predict(train_meta_features)
stacking_test_pred = best_meta_model.predict(test_meta_features)

# 计算Stacking模型在训练集上的整体性能
stacking_rmse = np.sqrt(mean_squared_error(y_train, stacking_train_pred))
print(f"Stacking集成模型在全体训练集上的RMSE: {stacking_rmse:.5f}")

# 模块4.3：与基准模型和简单平均集成的比较
print("\n" + "=" * 60)
print("模块4.3：性能对比分析")
print("=" * 60)

# 对比数据
comparison = {
    'Best Single Model (Lasso)': 0.12213,
    'Simple Averaging (Lasso+ElasticNet+LightGBM)': 0.11531,
    f'Stacking ({best_meta_name})': meta_results[best_meta_name][0]
}

print("模型性能对比 (交叉验证RMSE):")
for model, rmse in comparison.items():
    print(f"  {model:50} : {rmse:.5f}")

improvement_vs_single = (comparison['Best Single Model (Lasso)'] - comparison[f'Stacking ({best_meta_name})']) / \
                        comparison['Best Single Model (Lasso)'] * 100
improvement_vs_avg = (comparison['Simple Averaging (Lasso+ElasticNet+LightGBM)'] - comparison[
    f'Stacking ({best_meta_name})']) / comparison['Simple Averaging (Lasso+ElasticNet+LightGBM)'] * 100

print(f"\nStacking相对于最佳单模型(Lasso)的提升: {improvement_vs_single:.2f}%")
print(f"Stacking相对于简单平均集成的提升: {improvement_vs_avg:.2f}%")

# 为阶段五和阶段六保存Stacking预测结果
np.save('stacking_train_pred_log.npy', stacking_train_pred)
np.save('stacking_test_pred_log.npy', stacking_test_pred)
print("\nStacking预测结果已保存为NPY文件。")

# 可视化元特征与目标变量的关系（前两个基模型）
import matplotlib.pyplot as plt

#设置字体为系统自带的中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 设置中文字体
# 解决负号显示问题
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方块的问题

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].scatter(train_meta_features[:, 0], y_train, alpha=0.5)
axes[0].set_xlabel('Lasso 预测值 (元特征1)')
axes[0].set_ylabel('真实值 (Log SalePrice)')
axes[0].set_title('元特征1 vs 目标变量')
axes[1].scatter(train_meta_features[:, 2], y_train, alpha=0.5, color='green')
axes[1].set_xlabel('LightGBM 预测值 (元特征3)')
axes[1].set_ylabel('真实值 (Log SalePrice)')
axes[1].set_title('元特征3 vs 目标变量')
plt.tight_layout()
plt.show()
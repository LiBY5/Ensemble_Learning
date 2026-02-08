# Stacking集成深度优化：巩固优势，冲刺更低分
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import mean_squared_error
import warnings

warnings.filterwarnings('ignore')

print("=" * 60)
print("Stacking集成深度优化")
print("=" * 60)

# 1. 加载数据
X_train = pd.read_csv('X_train_processed.csv').values
y_train = pd.read_csv('y_train_log.csv').values.ravel()
X_test = pd.read_csv('X_test_processed.csv').values
print("数据加载完成。")

# 2. 定义优化后的基模型列表 (移除xgb，加入ElasticNet)
base_models = [
    ('lasso', Lasso(alpha=0.000534, max_iter=50000, random_state=42)),
    ('ridge', Ridge(alpha=10.0, random_state=42)),
    ('elasticnet', ElasticNet(alpha=0.0005, l1_ratio=0.9, max_iter=50000, random_state=42)),  # 新加入
    ('lightgbm', LGBMRegressor(n_estimators=216, learning_rate=0.101, max_depth=3,
                               num_leaves=37, subsample=0.98, colsample_bytree=0.94,
                               random_state=42, verbose=-1)),
    ('random_forest', RandomForestRegressor(n_estimators=234, max_depth=13,
                                            max_features=0.123, min_samples_split=4,
                                            random_state=42, n_jobs=-1)),
    ('catboost', CatBoostRegressor(iterations=279, depth=6, learning_rate=0.0766,
                                   l2_leaf_reg=8, random_state=42, verbose=0, thread_count=-1)),
]
print(f"使用 {len(base_models)} 个基模型: {[name for name, _ in base_models]}")

# 3. 第一层：生成OOF预测，并计算丰富的统计特征
n_folds = 7
kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

train_meta_features = np.zeros((X_train.shape[0], len(base_models)))
test_meta_features_list = []  # 存储各折的测试集预测，用于后续平均

print(f"\n使用 {n_folds} 折CV生成第一层预测及统计特征...")
for i, (name, model) in enumerate(base_models):
    print(f"  基模型: {name:15}", end="")
    test_fold_preds = []
    for train_idx, val_idx in kf.split(X_train, y_train):
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr = y_train[train_idx]
        # 注意：对于CatBoost等模型，clone可能需要特殊处理
        model.fit(X_tr, y_tr)
        train_meta_features[val_idx, i] = model.predict(X_val)
        test_fold_preds.append(model.predict(X_test))
    # 测试集预测取各折平均
    test_meta_avg = np.mean(test_fold_preds, axis=0)
    test_meta_features_list.append(test_meta_avg)
    model_rmse = np.sqrt(mean_squared_error(y_train, train_meta_features[:, i]))
    print(f"  OOF RMSE: {model_rmse:.5f}")

# 将测试集预测列表转为数组
test_meta_features = np.column_stack(test_meta_features_list)

# 4. 构建增强的第二层特征
print("\n构建增强的第二层特征（包含预测值统计量）...")


# 计算训练集和测试集的统计特征
def create_enhanced_meta_features(base_preds):
    """为基模型预测矩阵添加统计特征"""
    stats_features = np.column_stack([
        np.mean(base_preds, axis=1),  # 均值
        np.std(base_preds, axis=1),  # 标准差
        np.median(base_preds, axis=1),  # 中位数
        np.max(base_preds, axis=1) - np.min(base_preds, axis=1)  # 极差
    ])
    return np.hstack([base_preds, stats_features])


X_train_meta_enhanced = create_enhanced_meta_features(train_meta_features)
X_test_meta_enhanced = create_enhanced_meta_features(test_meta_features)
print(f"增强后的元特征形状: {X_train_meta_enhanced.shape}")

# 5. 元模型选择与调优
print("\n--- 元模型选择与调优 ---")
# 准备用于元模型调优的数据
# 这里我们使用增强后的元特征，但不拼接原始特征，先专注于优化元模型本身
meta_X_train = X_train_meta_enhanced
meta_X_test = X_test_meta_enhanced

# 尝试两种元模型：Ridge 和 Lasso
param_grid = {'alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]}
best_meta_model = None
best_meta_name = ""
best_score = float('inf')

for MetaModel, name in [(Ridge(random_state=42), "Ridge"), (Lasso(max_iter=10000, random_state=42), "Lasso")]:
    print(f"\n正在调优元模型: {name}")
    grid_search = GridSearchCV(MetaModel, param_grid, cv=5,
                               scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(meta_X_train, y_train)
    cv_rmse = np.sqrt(-grid_search.best_score_)
    print(f"  最佳 alpha: {grid_search.best_params_['alpha']}")
    print(f"  最佳CV RMSE: {cv_rmse:.5f}")

    if cv_rmse < best_score:
        best_score = cv_rmse
        best_meta_model = grid_search.best_estimator_
        best_meta_name = name

print(f"\n✅ 选择的最佳元模型: {best_meta_name} (alpha={best_meta_model.alpha}), CV RMSE: {best_score:.5f}")

# 6. 使用最佳元模型进行最终训练和预测
print("\n使用最佳配置训练最终Stacking模型...")
best_meta_model.fit(meta_X_train, y_train)
final_train_pred = best_meta_model.predict(meta_X_train)
final_test_pred_log = best_meta_model.predict(meta_X_test)

# 计算最终模型在训练集上的表现
final_rmse = np.sqrt(mean_squared_error(y_train, final_train_pred))
print(f"最终模型在全体训练集上的RMSE: {final_rmse:.5f}")

# 7. 生成提交文件
final_test_price = np.expm1(final_test_pred_log)
test_ids = pd.read_csv('test_ids.csv')['Id']
final_submission = pd.DataFrame({
    'Id': test_ids,
    'SalePrice': final_test_price
})
submission_path = f'stacking_enhanced_{best_meta_name.lower()}_submission.csv'
final_submission.to_csv(submission_path, index=False)
print(f"\n✅ 优化后的提交文件已保存: {submission_path}")
print("文件预览:")
print(final_submission.head())

# 8. 性能总结与建议
print("\n" + "=" * 60)
print("优化总结与后续步骤")
print("=" * 60)
print(f"历史最佳单模型 (Lasso) 公开分数: 0.13338")
print(f"前次两层Stacking公开分数: 0.12699")
print(f"本次优化Stacking模型CV RMSE: {best_score:.5f}")
print(f"\n建议：")
print(f"1. 立即将 '{submission_path}' 提交至Kaggle，验证优化效果。")
print(f"2. 若分数低于0.12699，则优化成功；可继续尝试拼接部分重要原始特征。")
print(f"3. 若分数未降低，则需分析过拟合，尝试增强正则化（增大alpha）或减少基模型数量。")
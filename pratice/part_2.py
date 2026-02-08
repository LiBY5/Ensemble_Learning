# Stacking集成优化 - 第三步：引入新模型与两层Stacking
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, Lasso
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold, RandomizedSearchCV
from scipy.stats import randint, uniform, loguniform
from sklearn.metrics import mean_squared_error
import warnings

warnings.filterwarnings('ignore')

print("=" * 60)
print("Stacking集成优化 - 第三步：引入新模型与两层Stacking")
print("=" * 60)

# 1. 加载数据
X_train = pd.read_csv('X_train_processed.csv')
y_train = pd.read_csv('y_train_log.csv').values.ravel()
X_test = pd.read_csv('X_test_processed.csv')
feature_names = X_train.columns.tolist()
X_train = X_train.values
X_test = X_test.values

print("数据加载完成，开始第三步优化。")

# 2. 对未充分调优的基模型进行调优 (RandomForest, XGBoost, 并引入CatBoost)
print("\n--- 步骤1: 对关键基模型进行调优 ---")


# 由于调优耗时，我们这里进行简化版的随机搜索，并限制迭代次数
def quick_tune(model, param_dist, X, y, model_name, n_iter=10):
    print(f"  正在调优 {model_name}...")
    search = RandomizedSearchCV(
        model, param_dist, n_iter=n_iter, cv=3,
        scoring='neg_mean_squared_error', random_state=42, n_jobs=-1
    )
    search.fit(X, y)
    best_rmse = np.sqrt(-search.best_score_)
    print(f"    最佳参数: {search.best_params_}")
    print(f"    最佳RMSE: {best_rmse:.5f}")
    return search.best_estimator_


# 2.1 调优 RandomForest
rf = RandomForestRegressor(random_state=42, n_jobs=-1)
rf_param_dist = {
    'n_estimators': randint(100, 300),
    'max_depth': randint(5, 15),
    'min_samples_split': randint(2, 10),
    'max_features': uniform(0.1, 0.5)  # 限制特征比例，增加随机性
}
best_rf = quick_tune(rf, rf_param_dist, X_train, y_train, "RandomForest")

# 2.2 调优 XGBoost
xgb = XGBRegressor(random_state=42, n_jobs=-1, verbosity=0)
xgb_param_dist = {
    'n_estimators': randint(100, 300),
    'max_depth': randint(3, 8),
    'learning_rate': loguniform(0.01, 0.3),
    'subsample': uniform(0.6, 0.4),
    'colsample_bytree': uniform(0.6, 0.4)
}
best_xgb = quick_tune(xgb, xgb_param_dist, X_train, y_train, "XGBoost")

# 2.3 引入并调优 CatBoost
print("  正在调优 CatBoost...")
# CatBoost调优稍慢，我们使用较少参数
cb = CatBoostRegressor(random_state=42, verbose=0, thread_count=-1)
cb_param_dist = {
    'iterations': randint(100, 300),
    'depth': randint(4, 8),
    'learning_rate': loguniform(0.01, 0.3),
    'l2_leaf_reg': randint(1, 10)
}
best_cb = quick_tune(cb, cb_param_dist, X_train, y_train, "CatBoost")

# 2.4 使用之前调优好的Lasso, Ridge, LightGBM
best_lasso = Lasso(alpha=0.000534, max_iter=50000, random_state=42)
best_ridge = Ridge(alpha=10.0, random_state=42)
best_lgb = LGBMRegressor(
    n_estimators=216, learning_rate=0.101, max_depth=3,
    num_leaves=37, subsample=0.98, colsample_bytree=0.94,
    random_state=42, verbose=-1
)

# 3. 构建强大的第一层基模型列表
base_models = [
    ('lasso', best_lasso),
    ('ridge', best_ridge),
    ('lightgbm', best_lgb),
    ('random_forest', best_rf),
    ('xgboost', best_xgb),
    ('catboost', best_cb)  # 新增
]
print(f"\n第一层将使用 {len(base_models)} 个调优后的基模型。")

# 4. 实现两层Stacking
print("\n--- 步骤2: 训练两层Stacking集成 ---")

# 4.1 第一层：生成OOF预测和测试集预测
n_folds = 7
kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

train_meta_features = np.zeros((X_train.shape[0], len(base_models)))
test_meta_features = np.zeros((X_test.shape[0], len(base_models)))

print(f"使用 {n_folds} 折CV生成第一层预测 (元特征)...")
for i, (name, model) in enumerate(base_models):
    print(f"  基模型: {name:15}", end="")
    test_fold_preds = []
    for train_idx, val_idx in kf.split(X_train, y_train):
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr = y_train[train_idx]
        model_clone = model.__class__(**model.get_params()) if hasattr(model, 'get_params') else model
        model_clone.fit(X_tr, y_tr)
        train_meta_features[val_idx, i] = model_clone.predict(X_val)
        test_fold_preds.append(model_clone.predict(X_test))
    # 测试集预测取各折平均
    test_meta_features[:, i] = np.mean(test_fold_preds, axis=0)
    model_rmse = np.sqrt(mean_squared_error(y_train, train_meta_features[:, i]))
    print(f"  OOF RMSE: {model_rmse:.5f}")

print(f"第一层元特征形状: {train_meta_features.shape}")

# 4.2 选择最重要的原始特征，与第一层预测拼接
print("\n--- 步骤3: 选择重要原始特征，构建第二层特征 ---")
# 使用Lasso的特征选择能力，找出最重要的原始特征
selector = Lasso(alpha=0.0005, max_iter=10000, random_state=42)
selector.fit(X_train, y_train)
# 获取系数绝对值大于阈值的特征索引
important_feature_idx = np.where(np.abs(selector.coef_) > 1e-4)[0]
print(f"从 {X_train.shape[1]} 个原始特征中选择了 {len(important_feature_idx)} 个重要特征。")
if len(important_feature_idx) > 20:  # 如果太多，只取前20个
    coef_abs = np.abs(selector.coef_[important_feature_idx])
    top_idx = np.argsort(coef_abs)[-20:]
    important_feature_idx = important_feature_idx[top_idx]
    print(f"保留最重要的 {len(important_feature_idx)} 个特征用于第二层。")

# 构建第二层特征 = [第一层预测, 重要原始特征]
X_train_layer2 = np.hstack([train_meta_features, X_train[:, important_feature_idx]])
X_test_layer2 = np.hstack([test_meta_features, X_test[:, important_feature_idx]])
print(f"第二层特征矩阵形状: {X_train_layer2.shape}")

# 4.3 第二层：训练元模型
print("\n--- 步骤4: 训练第二层元模型 ---")
# 使用简单的Ridge回归，避免过拟合
meta_model = Ridge(alpha=1.0, random_state=42)

# 评估两层Stacking性能
print("评估两层Stacking性能 (5折CV)...")
cv_scores = []
for train_idx, val_idx in kf.split(X_train_layer2, y_train):
    if len(train_idx) < n_folds:  # 确保训练集足够大
        continue
    X_tr2, X_val2 = X_train_layer2[train_idx], X_train_layer2[val_idx]
    y_tr, y_val = y_train[train_idx], y_train[val_idx]
    meta_model.fit(X_tr2, y_tr)
    val_pred = meta_model.predict(X_val2)
    score = np.sqrt(mean_squared_error(y_val, val_pred))
    cv_scores.append(score)

mean_rmse = np.mean(cv_scores)
std_rmse = np.std(cv_scores)
print(f"两层Stacking 平均RMSE: {mean_rmse:.5f} (±{std_rmse:.5f})")

# 5. 性能最终对比与决策
print("\n" + "=" * 60)
print("最终性能对比")
print("=" * 60)
comparison = {
    'Best Single Model (Lasso)': 0.10417,
    'Step1 Optimized Stacking': 0.11113,
    'Step3 Two-Layer Stacking': mean_rmse
}

print(f"{'Model':<35} | {'RMSE':<10} | {'Improvement vs Lasso':<20}")
print("-" * 70)
for model, rmse in comparison.items():
    if model != 'Best Single Model (Lasso)':
        impr = (0.10417 - rmse) / 0.10417 * 100
        print(f"{model:<35} | {rmse:<10.5f} | {impr:>+6.2f}%")
    else:
        print(f"{model:<35} | {rmse:<10.5f} | {'(基准)':>20}")

# 6. 训练最终模型并生成预测
if mean_rmse < 0.11113:  # 如果优于第一步优化
    print(f"\n✅ 两层Stacking性能优于第一步优化。")
    if mean_rmse < 0.10417:
        print(f"🎉 突破！两层Stacking ({mean_rmse:.5f}) 首次超越最佳单模型Lasso (0.10417)!")
        improvement = (0.10417 - mean_rmse) / 0.10417 * 100
        print(f"   相对提升: {improvement:.2f}%")
    else:
        print(f"⚠️  虽未超越单模型，但优于之前所有Stacking变体。")

    print("\n训练最终两层Stacking模型用于测试集预测...")
    # 使用全部数据重新训练第一层（简化，实际应保存各折模型）
    for i, (name, model) in enumerate(base_models):
        model.fit(X_train, y_train)  # 在全量数据上训练
    # 生成最终测试集元特征
    test_meta_final = np.column_stack([model.predict(X_test) for _, model in base_models])
    X_test_final = np.hstack([test_meta_final, X_test[:, important_feature_idx]])
    # 在全量第二层特征上训练元模型
    meta_model_final = Ridge(alpha=1.0, random_state=42)
    meta_model_final.fit(X_train_layer2, y_train)
    # 预测
    test_pred_log = meta_model_final.predict(X_test_final)

    # 转换回原始房价并保存
    test_pred_price = np.expm1(test_pred_log)
    test_ids = pd.read_csv('test_ids.csv')['Id']
    final_submission = pd.DataFrame({
        'Id': test_ids,
        'SalePrice': test_pred_price
    })
    submission_path = 'Prediction.csv'
    final_submission.to_csv(submission_path, index=False)
    print(f"最终提交文件已保存: {submission_path}")
    print("文件预览:")
    print(final_submission.head())
else:
    print(f"\n❌ 两层Stacking未带来提升。需要重新评估优化策略。")

# 7. 基模型贡献度分析
print("\n--- 基模型贡献度分析 ---")
# 通过元模型的系数绝对值，分析每个基模型预测的重要性
if hasattr(meta_model, 'coef_'):
    meta_coef = meta_model.coef_
    n_base_models = len(base_models)
    base_model_coef = np.abs(meta_coef[:n_base_models])
    print("第二层元模型赋予各基模型预测的权重 (绝对值):")
    for i, (name, _) in enumerate(base_models):
        print(f"  {name:15}: {base_model_coef[i]:.4f}")
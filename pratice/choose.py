from sklearn.ensemble import StackingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge, Lasso
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score, KFold
import numpy as np
import pandas as pd


# 1. 定义调优后的基模型
# 请将您阶段五找到的最佳参数填入
estimators = [
    ('lasso', Lasso(alpha=0.000534, max_iter=20000, random_state=42)),
    ('ridge', Ridge(alpha=10.0, random_state=42)),
    ('lightgbm', LGBMRegressor(n_estimators=216, learning_rate=0.101, max_depth=3,
                                num_leaves=37, subsample=0.98, colsample_bytree=0.94,
                                random_state=42, verbose=-1)),
    ('random_forest', RandomForestRegressor(n_estimators=100, max_features=0.3,
                                            random_state=42, n_jobs=-1)),
    ('xgboost', XGBRegressor(n_estimators=100, learning_rate=0.05,
                             max_depth=3, random_state=42, n_jobs=-1, verbosity=0))
]

# 2. 定义元模型
meta_model = Ridge(alpha=1.0, random_state=42)

# 3. 创建Stacking集成器
# passthrough=False 表示元模型只接收基模型的预测，不接收原始特征
stacking_regressor = StackingRegressor(
    estimators=estimators,
    final_estimator=meta_model,
    cv=7,  # 增加折数
    passthrough=False,
    n_jobs=-1
)

# 4. 在训练集上进行交叉验证评估
X_train = pd.read_csv('X_train_processed.csv').values
y_train = pd.read_csv('y_train_log.csv').values.ravel()

kf = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(stacking_regressor, X_train, y_train,
                         scoring='neg_mean_squared_error', cv=kf, n_jobs=-1)
stacking_rmse = np.sqrt(-scores.mean())
print(f"优化后Stacking集成 (5折CV) 平均RMSE: {stacking_rmse:.5f}")

# 5. 训练最终模型并预测测试集
stacking_regressor.fit(X_train, y_train)
X_test = pd.read_csv('X_test_processed.csv').values
test_pred_log = stacking_regressor.predict(X_test)

# 6. 保存预测结果
np.save('stacking_optimized_pred_log.npy', test_pred_log)
print("优化后的Stacking预测已保存。")
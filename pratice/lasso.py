# 调优后的Lasso单模型预测与提交
import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("调优后Lasso单模型 - 训练、预测与提交")
print("="*60)

# 1. 加载预处理后的数据
print("1. 加载预处理数据...")
X_train = pd.read_csv('X_train_processed.csv')
y_train_log = pd.read_csv('y_train_log.csv').values.ravel()  # 对数变换后的目标变量
X_test = pd.read_csv('X_test_processed.csv')

print(f"   训练集特征形状: {X_train.shape}")
print(f"   训练集目标形状: {y_train_log.shape}")
print(f"   测试集特征形状: {X_test.shape}")

# 2. 定义并使用调优后的Lasso模型
print("\n2. 初始化与训练调优后的Lasso模型...")
# 最佳参数来自之前的随机搜索: alpha=0.0005342937261279772
best_alpha = 0.0005342937261279772
lasso_model = Lasso(alpha=best_alpha, max_iter=50000, random_state=42)

# 训练模型
lasso_model.fit(X_train, y_train_log)
print(f"   模型训练完成。使用的正则化强度 alpha = {best_alpha:.6f}")

# 3. 在训练集上进行交叉验证或简单评估（可选）
print("\n3. 模型性能快速评估...")
# 使用训练集进行简单拟合评估（注意：这会低估真实误差）
train_pred_log = lasso_model.predict(X_train)
train_rmse = np.sqrt(mean_squared_error(y_train_log, train_pred_log))
print(f"   在全体训练集上的RMSE (对数空间): {train_rmse:.5f}")
print("   *注意：此分数为拟合分数，通常低于交叉验证分数。")

# 4. 对测试集进行预测
print("\n4. 对测试集进行预测...")
test_pred_log = lasso_model.predict(X_test)
print(f"   测试集预测完成。预测值范围 (对数空间): [{test_pred_log.min():.2f}, {test_pred_log.max():.2f}]")

# 5. 将预测值从对数空间转换回原始房价
print("\n5. 将预测值转换回原始房价...")
# 我们之前对目标变量使用了 np.log1p，现在使用 np.expm1 进行逆变换
test_pred_price = np.expm1(test_pred_log)
print(f"   转换后房价范围: ${test_pred_price.min():.2f} - ${test_pred_price.max():.2f}")
print(f"   房价中位数: ${np.median(test_pred_price):.2f}")
print(f"   房价平均值: ${test_pred_price.mean():.2f}")

# 6. 生成提交文件
print("\n6. 生成Kaggle提交文件...")
# 加载测试集ID
test_ids = pd.read_csv('test_ids.csv')['Id']
submission_df = pd.DataFrame({
    'Id': test_ids,
    'SalePrice': test_pred_price
})

# 保存提交文件
submission_file_path = 'lasso_tuned_submission.csv'
submission_df.to_csv(submission_file_path, index=False)
print(f"   提交文件已保存: {submission_file_path}")
print("\n   文件前5行预览:")
print(submission_df.head())

# 7. 与历史模型性能对比提示
print("\n" + "="*60)
print("对比与建议")
print("="*60)
print("您的模型历史性能参考 (交叉验证RMSE, 对数空间):")
print(f"  - 调优后Lasso单模型 (本次): ~0.10417 (来自之前5折CV)")
print(f"  - 两层Stacking集成: 0.11042 (来自之前5折CV)")
print(f"  - 首次提交 (Stacking) 公开分数: RMSLE = 0.12699")
print("\n【下一步建议】:")
print(f"1. 请将生成的 '{submission_file_path}' 提交至Kaggle。")
print("2. 对比Lasso单模型与Stacking集成的公开分数 (RMSLE)。")
print("3. 根据线上分数决定后续优化核心:")
print("   - 若Lasso分数更优: 深入优化特征工程与线性模型。")
print("   - 若Stacking分数更优: 继续深化集成学习策略。")
print("   - 若分数接近: 可尝试将两者结果进行加权融合。")

# 8. 保存模型预测结果以备后续融合使用（可选）
print("\n8. 保存Lasso预测结果供后续集成分析...")
np.save('lasso_tuned_test_pred_log.npy', test_pred_log)
np.save('lasso_tuned_train_pred_log.npy', train_pred_log)
print("   Lasso预测结果已保存为NPY文件。")
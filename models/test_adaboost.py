"""测试AdaBoost实现"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier as SklearnAdaBoost
from sklearn.metrics import accuracy_score, mean_squared_error
import os

# 确保目录存在
os.makedirs('../results/figures', exist_ok=True)

# 导入我们的实现
from models.boosting import AdaBoostClassifier, AdaBoostRegressor

def test_adaboost_classifier():
    """测试AdaBoost分类器"""
    print("="*60)
    print("AdaBoost分类器测试")
    print("="*60)

    # 生成二分类数据
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_classes=2,
        random_state=42
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"数据集信息:")
    print(f"  训练集大小: {X_train.shape}")
    print(f"  测试集大小: {X_test.shape}")
    print(f"  类别分布: {np.bincount(y_train)}")

    # 我们的实现
    our_adaboost = AdaBoostClassifier(
        base_estimator=DecisionTreeClassifier(max_depth=1),
        n_estimators=50,
        learning_rate=1.0,
        algorithm='SAMME',
        random_state=42
    )

    # sklearn的实现
    sklearn_adaboost = SklearnAdaBoost(
        estimator=DecisionTreeClassifier(max_depth=1),
        n_estimators=50,
        learning_rate=1.0,
        algorithm='SAMME',
        random_state=42
    )

    print("\n" + "="*60)
    print("训练我们的AdaBoost...")
    our_adaboost.fit(X_train, y_train)
    our_pred = our_adaboost.predict(X_test)
    our_acc = accuracy_score(y_test, our_pred)

    print(f"\n我们的实现结果:")
    print(f"  测试准确率: {our_acc:.4f}")
    print(f"  实际使用的基学习器数量: {our_adaboost.n_estimators}")
    print(f"  基学习器权重范围: [{our_adaboost.estimator_weights_[:our_adaboost.n_estimators].min():.4f}, "
          f"{our_adaboost.estimator_weights_[:our_adaboost.n_estimators].max():.4f}]")
    print(f"  基学习器错误率范围: [{our_adaboost.estimator_errors_[:our_adaboost.n_estimators].min():.4f}, "
          f"{our_adaboost.estimator_errors_[:our_adaboost.n_estimators].max():.4f}]")

    print("\n" + "="*60)
    print("训练sklearn的AdaBoost...")
    sklearn_adaboost.fit(X_train, y_train)
    sklearn_pred = sklearn_adaboost.predict(X_test)
    sklearn_acc = accuracy_score(y_test, sklearn_pred)

    print(f"\nsklearn实现结果:")
    print(f"  测试准确率: {sklearn_acc:.4f}")
    print(f"  实际使用的基学习器数量: {len(sklearn_adaboost.estimators_)}")

    # 可视化训练过程
    visualize_training_process(our_adaboost, X_train, y_train, X_test, y_test)

    return our_acc, sklearn_acc, our_adaboost, sklearn_adaboost

def visualize_training_process(model, X_train, y_train, X_test, y_test):
    """可视化训练过程"""
    print("\n" + "="*60)
    print("可视化训练过程...")

    train_errors = []
    test_errors = []

    # 获取每个阶段的预测
    for i, y_pred in enumerate(model.staged_predict(X_train), 1):
        train_errors.append(1 - accuracy_score(y_train, y_pred))

    for i, y_pred in enumerate(model.staged_predict(X_test), 1):
        test_errors.append(1 - accuracy_score(y_test, y_pred))

    # 绘制误差曲线
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.plot(range(1, len(train_errors) + 1), train_errors,
             'b-', label='训练误差', linewidth=2)
    plt.plot(range(1, len(test_errors) + 1), test_errors,
             'r-', label='测试误差', linewidth=2)
    plt.xlabel('基学习器数量')
    plt.ylabel('误差')
    plt.title('AdaBoost训练过程 - 误差曲线')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 标记最佳测试误差
    best_idx = np.argmin(test_errors)
    plt.axvline(x=best_idx + 1, color='g', linestyle='--',
                label=f'最佳: {best_idx+1}棵树')
    plt.scatter(best_idx + 1, test_errors[best_idx],
                color='g', s=100, zorder=5)

    # 绘制基学习器权重
    plt.subplot(2, 2, 2)
    plt.bar(range(1, model.n_estimators + 1),
            model.estimator_weights_[:model.n_estimators])
    plt.xlabel('基学习器索引')
    plt.ylabel('权重')
    plt.title('基学习器权重分布')
    plt.grid(True, alpha=0.3, axis='y')

    # 绘制基学习器错误率
    plt.subplot(2, 2, 3)
    plt.bar(range(1, model.n_estimators + 1),
            model.estimator_errors_[:model.n_estimators])
    plt.xlabel('基学习器索引')
    plt.ylabel('错误率')
    plt.title('基学习器错误率')
    plt.grid(True, alpha=0.3, axis='y')

    # 绘制训练过程中的权重变化
    plt.subplot(2, 2, 4)
    plt.semilogy(range(1, len(train_errors) + 1), train_errors,
                 'b-', label='训练误差', linewidth=2)
    plt.semilogy(range(1, len(test_errors) + 1), test_errors,
                 'r-', label='测试误差', linewidth=2)
    plt.xlabel('基学习器数量')
    plt.ylabel('误差（对数尺度）')
    plt.title('AdaBoost训练过程 - 对数尺度')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    # 设置字体为系统自带的中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'KaiTi', 'FangSong']  # 设置中文字体
    # 解决负号显示问题
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方块的问题
    plt.savefig('../results/figures/day3_adaboost_training_process.png', dpi=150)
    plt.show()

    print(f"\n训练过程分析:")
    print(f"  最终训练误差: {train_errors[-1]:.4f}")
    print(f"  最终测试误差: {test_errors[-1]:.4f}")
    print(f"  最佳测试误差在第{best_idx+1}棵树: {test_errors[best_idx]:.4f}")
    print(f"  过拟合程度（测试误差-训练误差）: {test_errors[-1] - train_errors[-1]:.4f}")

def test_adaboost_regressor():
    """测试AdaBoost回归器"""
    print("\n" + "="*60)
    print("AdaBoost回归器测试")
    print("="*60)

    # 生成回归数据
    X, y = make_regression(
        n_samples=500,
        n_features=10,
        n_informative=8,
        noise=20,
        random_state=42
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"数据集信息:")
    print(f"  训练集大小: {X_train.shape}")
    print(f"  测试集大小: {X_test.shape}")
    print(f"  目标值范围: [{y.min():.2f}, {y.max():.2f}]")

    # 我们的实现
    our_adaboost_reg = AdaBoostRegressor(
        n_estimators=30,
        learning_rate=0.1,
        loss='square',
        random_state=42
    )

    print("\n训练我们的AdaBoost回归器...")
    our_adaboost_reg.fit(X_train, y_train)
    our_pred = our_adaboost_reg.predict(X_test)
    our_mse = mean_squared_error(y_test, our_pred)

    print(f"\n回归结果:")
    print(f"  测试MSE: {our_mse:.4f}")
    print(f"  RMSE: {np.sqrt(our_mse):.4f}")
    print(f"  实际使用的基学习器数量: {len(our_adaboost_reg.estimators_)}")

    # 可视化预测结果
    plt.figure(figsize=(10, 8))

    plt.subplot(2, 2, 1)
    plt.scatter(y_test, our_pred, alpha=0.6, edgecolors='k', linewidth=0.5)

    # 计算回归线
    z = np.polyfit(y_test, our_pred, 1)
    p = np.poly1d(z)
    x_range = np.linspace(y_test.min(), y_test.max(), 100)
    plt.plot(x_range, p(x_range), 'r--', linewidth=2,
             label=f'y = {z[0]:.2f}x + {z[1]:.2f}')

    # 理想对角线
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
             'g-', linewidth=2, alpha=0.5, label='理想预测')

    plt.xlabel('真实值')
    plt.ylabel('预测值')
    plt.title('AdaBoost回归预测结果')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 绘制残差图
    plt.subplot(2, 2, 2)
    residuals = y_test - our_pred
    plt.scatter(our_pred, residuals, alpha=0.6, edgecolors='k', linewidth=0.5)
    plt.axhline(y=0, color='r', linestyle='--', linewidth=2)
    plt.xlabel('预测值')
    plt.ylabel('残差')
    plt.title('残差图')
    plt.grid(True, alpha=0.3)

    # 绘制误差分布
    plt.subplot(2, 2, 3)
    plt.hist(residuals, bins=30, edgecolor='black', alpha=0.7)
    plt.xlabel('残差')
    plt.ylabel('频率')
    plt.title('残差分布')
    plt.grid(True, alpha=0.3)

    # 绘制基学习器权重
    plt.subplot(2, 2, 4)
    plt.bar(range(1, len(our_adaboost_reg.estimator_weights_) + 1),
            our_adaboost_reg.estimator_weights_)
    plt.xlabel('基学习器索引')
    plt.ylabel('权重')
    plt.title('回归基学习器权重分布')
    plt.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    # 设置字体为系统自带的中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'KaiTi', 'FangSong']  # 设置中文字体
    # 解决负号显示问题
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方块的问题
    plt.savefig('../results/figures/day3_adaboost_regression.png', dpi=150)
    plt.show()

    print(f"\n残差统计:")
    print(f"  残差均值: {residuals.mean():.4f}")
    print(f"  残差标准差: {residuals.std():.4f}")
    print(f"  最大残差: {residuals.max():.4f}")
    print(f"  最小残差: {residuals.min():.4f}")

    return our_mse

if __name__ == "__main__":
    # 测试分类器
    our_acc, sklearn_acc, our_model, sklearn_model = test_adaboost_classifier()

    print("\n" + "="*60)
    print("模型对比分析")
    print("="*60)
    print(f"我们的实现准确率: {our_acc:.4f}")
    print(f"sklearn实现准确率: {sklearn_acc:.4f}")
    print(f"准确率差异: {abs(our_acc - sklearn_acc):.4f}")
    print(f"相对差异: {abs(our_acc - sklearn_acc) / sklearn_acc * 100:.2f}%")

    # 测试回归器
    our_mse = test_adaboost_regressor()

    print("\n" + "="*60)
    print("测试完成!")
    print("="*60)
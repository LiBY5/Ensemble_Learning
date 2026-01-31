"""简单测试AdaBoost实现"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier as SklearnAdaBoost
from sklearn.metrics import accuracy_score, mean_squared_error
import sys
import os

# 添加当前目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from models.boosting import AdaBoostClassifier, AdaBoostRegressor
    print("✓ 成功导入AdaBoostClassifier和AdaBoostRegressor")
except ImportError as e:
    print(f"✗ 导入失败: {e}")
    print(f"当前工作目录: {os.getcwd()}")
    print(f"Python路径: {sys.path}")
    sys.exit(1)

def test_classifier_simple():
    """简单测试分类器"""
    print("\n" + "="*60)
    print("测试AdaBoost分类器")
    print("="*60)

    # 生成简单的二分类数据
    X, y = make_classification(
        n_samples=200,
        n_features=5,
        n_informative=3,
        n_classes=2,
        random_state=42
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 我们的实现
    print("\n1. 训练我们的AdaBoost实现...")
    our_adaboost = AdaBoostClassifier(
        base_estimator=DecisionTreeClassifier(max_depth=1, random_state=42),
        n_estimators=20,
        learning_rate=0.8,
        random_state=42
    )

    our_adaboost.fit(X_train, y_train)
    our_pred = our_adaboost.predict(X_test)
    our_acc = accuracy_score(y_test, our_pred)

    print(f"   准确率: {our_acc:.4f}")
    print(f"   使用基学习器数量: {our_adaboost.n_estimators}")

    # sklearn的实现
    print("\n2. 训练sklearn的AdaBoost...")
    sklearn_adaboost = SklearnAdaBoost(
        estimator=DecisionTreeClassifier(max_depth=1, random_state=42),
        n_estimators=20,
        learning_rate=0.8,
        random_state=42
    )

    sklearn_adaboost.fit(X_train, y_train)
    sklearn_pred = sklearn_adaboost.predict(X_test)
    sklearn_acc = accuracy_score(y_test, sklearn_pred)

    print(f"   准确率: {sklearn_acc:.4f}")

    # 比较
    print(f"\n3. 结果比较:")
    print(f"   我们的实现: {our_acc:.4f}")
    print(f"   sklearn实现: {sklearn_acc:.4f}")
    print(f"   差异: {abs(our_acc - sklearn_acc):.4f}")

    if abs(our_acc - sklearn_acc) < 0.1:
        print("   ✓ 性能接近，实现正确！")
    else:
        print("   ⚠ 性能差异较大")

    return our_acc, sklearn_acc

def test_regressor_simple():
    """测试回归器"""
    print("\n" + "="*60)
    print("测试AdaBoost回归器")
    print("="*60)

    # 生成简单的回归数据
    X, y = make_regression(
        n_samples=200,
        n_features=5,
        n_informative=3,
        noise=10,
        random_state=42
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 我们的实现
    print("\n1. 训练我们的AdaBoost回归器...")
    our_adaboost_reg = AdaBoostRegressor(
        n_estimators=20,
        learning_rate=0.1,
        loss='square',
        random_state=42
    )

    our_adaboost_reg.fit(X_train, y_train)
    our_pred = our_adaboost_reg.predict(X_test)
    our_mse = mean_squared_error(y_test, our_pred)
    our_rmse = np.sqrt(our_mse)

    print(f"   MSE: {our_mse:.4f}")
    print(f"   RMSE: {our_rmse:.4f}")
    print(f"   R²分数: {our_adaboost_reg.score(X_test, y_test):.4f}")

    # 可视化
    print("\n2. 可视化预测结果...")
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, our_pred, alpha=0.5, label='预测点')

    # 完美预测线
    min_val = min(y_test.min(), our_pred.min())
    max_val = max(y_test.max(), our_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='完美预测')

    plt.xlabel('真实值', fontsize=12)
    plt.ylabel('预测值', fontsize=12)
    plt.title(f'AdaBoost回归预测 (RMSE={our_rmse:.2f})', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 添加残差分布子图
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    ax_inset = inset_axes(plt.gca(), width="30%", height="30%", loc='upper left')
    residuals = y_test - our_pred
    ax_inset.hist(residuals, bins=20, alpha=0.7, color='orange', edgecolor='black')
    ax_inset.set_xlabel('残差', fontsize=8)
    ax_inset.set_ylabel('频数', fontsize=8)
    ax_inset.set_title('残差分布', fontsize=9)
    ax_inset.grid(True, alpha=0.3)

    plt.tight_layout()

    # 保存图片
    os.makedirs('results/figures', exist_ok=True)
    plt.savefig('results/figures/day3_adaboost_regression_simple.png', dpi=150, bbox_inches='tight')
    plt.show()

    return our_mse

if __name__ == "__main__":
    print("开始AdaBoost测试...")
    print(f"工作目录: {os.getcwd()}")

    # 测试分类器
    our_acc, sklearn_acc = test_classifier_simple()

    # 测试回归器
    our_mse = test_regressor_simple()

    print("\n" + "="*60)
    print("所有测试完成！")
    print("="*60)
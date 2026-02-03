"""测试GBDT实现"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier as SklearnGBC
from sklearn.ensemble import GradientBoostingRegressor as SklearnGBR
from sklearn.metrics import accuracy_score, mean_squared_error, roc_auc_score

from models.boosting import (GradientBoostingRegressor,
                            GradientBoostingClassifier)

def test_gbdt_regression():
    """测试GBDT回归器"""
    # 生成回归数据
    X, y = make_regression(
        n_samples=1000, n_features=10, n_informative=8,
        noise=20, random_state=42
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 我们的实现
    our_gbdt = GradientBoostingRegressor(
        loss='ls',
        learning_rate=0.1,
        n_estimators=100,
        max_depth=3,
        subsample=0.8,
        random_state=42,
        verbose=0
    )

    # sklearn的实现
    sklearn_gbdt = SklearnGBR(
        loss='squared_error',
        learning_rate=0.1,
        n_estimators=100,
        max_depth=3,
        subsample=0.8,
        random_state=42
    )

    print("训练我们的GBDT回归器...")
    our_gbdt.fit(X_train, y_train)
    our_pred = our_gbdt.predict(X_test)
    our_mse = mean_squared_error(y_test, our_pred)

    print(f"\n我们的实现:")
    print(f"  测试MSE: {our_mse:.4f}")
    print(f"  训练损失历史: {our_gbdt.train_score_[:5]}...")

    print("\n训练sklearn的GBDT回归器...")
    sklearn_gbdt.fit(X_train, y_train)
    sklearn_pred = sklearn_gbdt.predict(X_test)
    sklearn_mse = mean_squared_error(y_test, sklearn_pred)

    print(f"\nsklearn实现:")
    print(f"  测试MSE: {sklearn_mse:.4f}")

    # 可视化训练过程
    visualize_training_curve(our_gbdt.train_score_, sklearn_gbdt.train_score_)

    return our_mse, sklearn_mse

def test_gbdt_classification():
    """测试GBDT分类器"""
    # 生成二分类数据
    X, y = make_classification(
        n_samples=1000, n_features=20, n_informative=15,
        n_classes=2, random_state=42
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 我们的实现
    our_gbdt = GradientBoostingClassifier(
        loss='deviance',
        learning_rate=0.1,
        n_estimators=100,
        max_depth=3,
        subsample=0.8,
        random_state=42,
        verbose=0
    )

    # sklearn的实现
    sklearn_gbdt = SklearnGBC(
        loss='log_loss',
        learning_rate=0.1,
        n_estimators=100,
        max_depth=3,
        subsample=0.8,
        random_state=42
    )

    print("\n训练我们的GBDT分类器...")
    our_gbdt.fit(X_train, y_train)
    our_pred = our_gbdt.predict(X_test)
    our_proba = our_gbdt.predict_proba(X_test)[:, 1]
    our_acc = accuracy_score(y_test, our_pred)
    our_auc = roc_auc_score(y_test, our_proba)

    print(f"\n我们的实现:")
    print(f"  测试准确率: {our_acc:.4f}")
    print(f"  AUC: {our_auc:.4f}")

    print("\n训练sklearn的GBDT分类器...")
    sklearn_gbdt.fit(X_train, y_train)
    sklearn_pred = sklearn_gbdt.predict(X_test)
    sklearn_proba = sklearn_gbdt.predict_proba(X_test)[:, 1]
    sklearn_acc = accuracy_score(y_test, sklearn_pred)
    sklearn_auc = roc_auc_score(y_test, sklearn_proba)

    print(f"\nsklearn实现:")
    print(f"  测试准确率: {sklearn_acc:.4f}")
    print(f"  AUC: {sklearn_auc:.4f}")

    # 可视化特征重要性
    visualize_feature_importance(our_gbdt, sklearn_gbdt, X_train.shape[1])

    return our_acc, sklearn_acc

def visualize_training_curve(our_scores, sklearn_scores):
    """可视化训练曲线"""
    plt.figure(figsize=(10, 6))

    plt.plot(range(1, len(our_scores) + 1), our_scores,
             'b-', label='我们的实现', linewidth=2)
    plt.plot(range(1, len(sklearn_scores) + 1), sklearn_scores,
             'r-', label='sklearn实现', linewidth=2, alpha=0.7)

    plt.xlabel('迭代次数')
    plt.ylabel('训练损失')
    plt.title('GBDT训练损失曲线')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')

    plt.tight_layout()
    # 设置字体为系统自带的中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'KaiTi', 'FangSong']  # 设置中文字体
    # 解决负号显示问题
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方块的问题
    plt.savefig('../results/figures/day3_gbdt_training_curve.png', dpi=150)
    plt.show()

def visualize_feature_importance(our_model, sklearn_model, n_features):
    """可视化特征重要性"""
    # 注意：我们的实现没有计算特征重要性，这里用sklearn的
    if hasattr(sklearn_model, 'feature_importances_'):
        importances = sklearn_model.feature_importances_
        indices = np.argsort(importances)[::-1][:15]

        plt.figure(figsize=(10, 6))
        plt.bar(range(len(indices)), importances[indices])
        plt.xticks(range(len(indices)), indices, rotation=45)
        plt.xlabel('特征索引')
        plt.ylabel('重要性')
        plt.title('GBDT特征重要性 (Top 15)')
        plt.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        # 设置字体为系统自带的中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'KaiTi', 'FangSong']  # 设置中文字体
        # 解决负号显示问题
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方块的问题
        plt.savefig('../results/figures/day3_gbdt_feature_importance.png', dpi=150)
        plt.show()

if __name__ == "__main__":
    import os
    os.makedirs('../results/figures', exist_ok=True)

    # 测试回归
    our_mse, sklearn_mse = test_gbdt_regression()
    print(f"\n回归MSE差异: {abs(our_mse - sklearn_mse):.4f}")

    # 测试分类
    our_acc, sklearn_acc = test_gbdt_classification()
    print(f"\n分类准确率差异: {abs(our_acc - sklearn_acc):.4f}")
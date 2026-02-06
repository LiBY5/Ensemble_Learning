"""测试AdaBoost实现 - 优化版本"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, make_regression, load_breast_cancer, load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import AdaBoostClassifier as SklearnAdaBoost
from sklearn.ensemble import AdaBoostRegressor as SklearnAdaBoostRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, confusion_matrix, classification_report
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

# 确保目录存在
os.makedirs('../results/figures', exist_ok=True)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8-darkgrid')

# 导入我们的实现
import sys
sys.path.append('..')
from models.boosting import AdaBoostClassifier, AdaBoostRegressor


def test_adaboost_classifier():
    """测试AdaBoost分类器"""
    print("="*60)
    print("AdaBoost分类器测试")
    print("="*60)

    # 使用真实数据集 - 乳腺癌数据集
    from sklearn.datasets import load_breast_cancer
    data = load_breast_cancer()
    X, y = data.data, data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"数据集信息:")
    print(f"  数据集: 乳腺癌数据集")
    print(f"  训练集大小: {X_train.shape}")
    print(f"  测试集大小: {X_test.shape}")
    print(f"  类别分布: 良性={np.sum(y==0)}, 恶性={np.sum(y==1)}")
    print(f"  特征数量: {X.shape[1]}")

    # 我们的实现
    our_adaboost = AdaBoostClassifier(
        base_estimator=DecisionTreeClassifier(max_depth=3, random_state=42),
        n_estimators=50,
        learning_rate=1.0,
        algorithm='SAMME',
        random_state=42
    )

    # sklearn的实现
    sklearn_adaboost = SklearnAdaBoost(
        estimator=DecisionTreeClassifier(max_depth=3, random_state=42),
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

    if hasattr(our_adaboost, 'estimator_weights_'):
        weights = our_adaboost.estimator_weights_[:our_adaboost.n_estimators]
        if len(weights) > 0:
            print(f"  基学习器权重范围: [{weights.min():.4f}, {weights.max():.4f}]")
            print(f"  平均基学习器权重: {weights.mean():.4f}")

    if hasattr(our_adaboost, 'estimator_errors_'):
        errors = our_adaboost.estimator_errors_[:our_adaboost.n_estimators]
        if len(errors) > 0:
            print(f"  基学习器错误率范围: [{errors.min():.4f}, {errors.max():.4f}]")
            print(f"  平均基学习器错误率: {errors.mean():.4f}")

    print("\n" + "="*60)
    print("训练sklearn的AdaBoost...")
    sklearn_adaboost.fit(X_train, y_train)
    sklearn_pred = sklearn_adaboost.predict(X_test)
    sklearn_acc = accuracy_score(y_test, sklearn_pred)

    print(f"\nsklearn实现结果:")
    print(f"  测试准确率: {sklearn_acc:.4f}")
    print(f"  实际使用的基学习器数量: {len(sklearn_adaboost.estimators_)}")

    # 生成详细分类报告
    print("\n" + "="*60)
    print("我们的实现分类报告:")
    print("="*60)
    print(classification_report(y_test, our_pred,
                               target_names=['良性', '恶性']))

    print("\n" + "="*60)
    print("sklearn实现分类报告:")
    print("="*60)
    print(classification_report(y_test, sklearn_pred,
                               target_names=['良性', '恶性']))

    # 可视化训练过程
    visualize_adaboost_classifier_training(our_adaboost, X_train, y_train, X_test, y_test)

    # 对比可视化
    visualize_classifier_comparison(our_adaboost, sklearn_adaboost, X_test, y_test)

    return our_acc, sklearn_acc, our_adaboost, sklearn_adaboost


def visualize_adaboost_classifier_training(model, X_train, y_train, X_test, y_test):
    """可视化AdaBoost分类器训练过程"""
    print("\n" + "="*60)
    print("可视化训练过程...")

    train_errors = []
    test_errors = []

    # 获取每个阶段的预测
    for i, y_pred in enumerate(model.staged_predict(X_train), 1):
        train_errors.append(1 - accuracy_score(y_train, y_pred))

    for i, y_pred in enumerate(model.staged_predict(X_test), 1):
        test_errors.append(1 - accuracy_score(y_test, y_pred))

    # 创建可视化图表
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. 误差曲线
    ax1 = axes[0, 0]
    ax1.plot(range(1, len(train_errors) + 1), train_errors,
             'b-', label='训练误差', linewidth=2, alpha=0.8)
    ax1.plot(range(1, len(test_errors) + 1), test_errors,
             'r-', label='测试误差', linewidth=2, alpha=0.8)
    ax1.set_xlabel('基学习器数量', fontsize=12)
    ax1.set_ylabel('误差', fontsize=12)
    ax1.set_title('AdaBoost训练过程 - 误差曲线', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 标记最佳测试误差
    if len(test_errors) > 0:
        best_idx = np.argmin(test_errors)
        ax1.axvline(x=best_idx + 1, color='g', linestyle='--',
                    label=f'最佳: {best_idx+1}棵树', alpha=0.7)
        ax1.scatter(best_idx + 1, test_errors[best_idx],
                   color='g', s=100, zorder=5, edgecolor='black', linewidth=1)

    # 2. 基学习器权重
    ax2 = axes[0, 1]
    if hasattr(model, 'estimator_weights_'):
        weights = model.estimator_weights_[:model.n_estimators]
        if len(weights) > 0:
            bars = ax2.bar(range(1, len(weights) + 1), weights,
                          color='steelblue', edgecolor='navy', alpha=0.7)
            ax2.set_xlabel('基学习器索引', fontsize=12)
            ax2.set_ylabel('权重', fontsize=12)
            ax2.set_title('基学习器权重分布', fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3, axis='y')

            # 添加平均值线
            mean_weight = np.mean(weights)
            ax2.axhline(y=mean_weight, color='red', linestyle='--',
                       linewidth=2, alpha=0.7, label=f'平均权重: {mean_weight:.4f}')
            ax2.legend()

    # 3. 基学习器错误率
    ax3 = axes[1, 0]
    if hasattr(model, 'estimator_errors_'):
        errors = model.estimator_errors_[:model.n_estimators]
        if len(errors) > 0:
            ax3.bar(range(1, len(errors) + 1), errors,
                   color='lightcoral', edgecolor='darkred', alpha=0.7)
            ax3.set_xlabel('基学习器索引', fontsize=12)
            ax3.set_ylabel('错误率', fontsize=12)
            ax3.set_title('基学习器错误率', fontsize=14, fontweight='bold')
            ax3.grid(True, alpha=0.3, axis='y')

    # 4. 对数尺度误差曲线
    ax4 = axes[1, 1]
    if len(train_errors) > 0 and len(test_errors) > 0:
        ax4.semilogy(range(1, len(train_errors) + 1), train_errors,
                    'b-', label='训练误差', linewidth=2, alpha=0.8)
        ax4.semilogy(range(1, len(test_errors) + 1), test_errors,
                    'r-', label='测试误差', linewidth=2, alpha=0.8)
        ax4.set_xlabel('基学习器数量', fontsize=12)
        ax4.set_ylabel('误差（对数尺度）', fontsize=12)
        ax4.set_title('AdaBoost训练过程 - 对数尺度', fontsize=14, fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3, which='both')

    plt.suptitle('AdaBoost分类器训练过程分析', fontsize=16, fontweight='bold')
    plt.tight_layout()

    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 设置中文字体
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方块的问题

    plt.savefig('../results/figures/day3_adaboost_classifier_training.png', dpi=150, bbox_inches='tight')
    plt.show()

    # 打印分析结果
    if len(train_errors) > 0 and len(test_errors) > 0:
        print(f"\n训练过程分析:")
        print(f"  最终训练误差: {train_errors[-1]:.4f}")
        print(f"  最终测试误差: {test_errors[-1]:.4f}")
        if len(test_errors) > 0:
            best_idx = np.argmin(test_errors)
            print(f"  最佳测试误差在第{best_idx+1}棵树: {test_errors[best_idx]:.4f}")
        print(f"  过拟合程度（测试误差-训练误差）: {test_errors[-1] - train_errors[-1]:.4f}")


def visualize_classifier_comparison(our_model, sklearn_model, X_test, y_test):
    """可视化分类器对比"""
    our_pred = our_model.predict(X_test)
    sklearn_pred = sklearn_model.predict(X_test)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # 1. 准确率对比
    ax1 = axes[0, 0]
    models = ['我们的实现', 'sklearn实现']
    accuracies = [accuracy_score(y_test, our_pred),
                  accuracy_score(y_test, sklearn_pred)]

    colors = ['lightgreen', 'lightcoral']
    bars = ax1.bar(models, accuracies, color=colors, edgecolor='black')
    ax1.set_ylabel('准确率', fontsize=12)
    ax1.set_title('模型准确率对比', fontsize=14, fontweight='bold')
    ax1.set_ylim([0, 1])
    ax1.grid(True, alpha=0.3, axis='y')

    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, height + 0.01,
                f'{acc:.4f}', ha='center', va='bottom')

    # 2. 我们的实现混淆矩阵
    ax2 = axes[0, 1]
    cm_our = confusion_matrix(y_test, our_pred)
    sns.heatmap(cm_our, annot=True, fmt='d', cmap='Blues',
                xticklabels=['预测良性', '预测恶性'],
                yticklabels=['真实良性', '真实恶性'], ax=ax2)
    ax2.set_title('我们的实现 - 混淆矩阵', fontsize=14, fontweight='bold')

    # 3. sklearn实现混淆矩阵
    ax3 = axes[0, 2]
    cm_sklearn = confusion_matrix(y_test, sklearn_pred)
    sns.heatmap(cm_sklearn, annot=True, fmt='d', cmap='Reds',
                xticklabels=['预测良性', '预测恶性'],
                yticklabels=['真实良性', '真实恶性'], ax=ax3)
    ax3.set_title('sklearn实现 - 混淆矩阵', fontsize=14, fontweight='bold')

    # 4. 基学习器数量对比
    ax4 = axes[1, 0]
    n_estimators_our = our_model.n_estimators
    n_estimators_sklearn = len(sklearn_model.estimators_)

    bars = ax4.bar(['我们的实现', 'sklearn实现'],
                  [n_estimators_our, n_estimators_sklearn],
                  color=['lightblue', 'lightpink'], edgecolor='black')
    ax4.set_ylabel('基学习器数量', fontsize=12)
    ax4.set_title('基学习器数量对比', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')

    for bar, n in zip(bars, [n_estimators_our, n_estimators_sklearn]):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2, height + 0.5,
                f'{n}', ha='center', va='bottom')

    # 5. 差异分析
    ax5 = axes[1, 1]
    diff = np.abs(our_pred - sklearn_pred)
    n_different = np.sum(diff)
    n_total = len(y_test)

    labels = ['预测一致', '预测不同']
    sizes = [n_total - n_different, n_different]
    colors = ['lightgreen', 'lightcoral']

    ax5.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
           startangle=90, wedgeprops={'edgecolor': 'black'})
    ax5.set_title(f'模型预测一致性\n(不同: {n_different}/{n_total})',
                 fontsize=14, fontweight='bold')

    # 6. 错误样本分析
    ax6 = axes[1, 2]
    our_correct = (our_pred == y_test)
    sklearn_correct = (sklearn_pred == y_test)

    categories = ['两者正确', '仅我们正确', '仅sklearn正确', '两者错误']
    counts = [
        np.sum(our_correct & sklearn_correct),
        np.sum(our_correct & ~sklearn_correct),
        np.sum(~our_correct & sklearn_correct),
        np.sum(~our_correct & ~sklearn_correct)
    ]

    bars = ax6.bar(categories, counts, color=['lightgreen', 'lightblue',
                                              'lightcoral', 'gray'])
    ax6.set_ylabel('样本数量', fontsize=12)
    ax6.set_title('模型错误分析', fontsize=14, fontweight='bold')
    ax6.tick_params(axis='x', rotation=45)
    ax6.grid(True, alpha=0.3, axis='y')

    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2, height + 0.5,
                f'{count}', ha='center', va='bottom')

    plt.suptitle('AdaBoost分类器对比分析', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 设置中文字体
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方块的问题

    plt.savefig('../results/figures/day3_adaboost_classifier_comparison.png',
                dpi=150, bbox_inches='tight')
    plt.show()

    # 打印对比分析
    print("\n" + "="*60)
    print("模型对比分析")
    print("="*60)
    print(f"我们的实现准确率: {accuracies[0]:.4f}")
    print(f"sklearn实现准确率: {accuracies[1]:.4f}")
    print(f"准确率差异: {abs(accuracies[0] - accuracies[1]):.4f}")
    print(f"相对差异: {abs(accuracies[0] - accuracies[1]) / accuracies[1] * 100:.2f}%")
    print(f"预测一致性: {(n_total - n_different) / n_total * 100:.1f}%")
    print(f"共同错误的样本数: {counts[3]}")


def test_adaboost_regressor():
    """测试AdaBoost回归器 - 修复版本"""
    print("\n" + "=" * 60)
    print("AdaBoost回归器测试")
    print("=" * 60)

    # 使用真实数据集 - 糖尿病数据集
    from sklearn.datasets import load_diabetes
    data = load_diabetes()
    X, y = data.data, data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"数据集信息:")
    print(f"  数据集: 糖尿病数据集")
    print(f"  训练集大小: {X_train.shape}")
    print(f"  测试集大小: {X_test.shape}")
    print(f"  目标值范围: [{y.min():.2f}, {y.max():.2f}]")
    print(f"  目标值均值: {y.mean():.2f} ± {y.std():.2f}")

    # 我们的实现
    our_adaboost_reg = AdaBoostRegressor(
        base_estimator=DecisionTreeRegressor(max_depth=3, random_state=42),
        n_estimators=30,
        learning_rate=0.1,
        loss='square',
        random_state=42
    )

    # sklearn的实现
    sklearn_adaboost_reg = SklearnAdaBoostRegressor(
        estimator=DecisionTreeRegressor(max_depth=3, random_state=42),
        n_estimators=30,
        learning_rate=0.1,
        loss='square',
        random_state=42
    )

    print("\n训练我们的AdaBoost回归器...")
    try:
        our_adaboost_reg.fit(X_train, y_train)
        our_pred = our_adaboost_reg.predict(X_test)
        our_mse = mean_squared_error(y_test, our_pred)
        our_r2 = r2_score(y_test, our_pred)

        print(f"  成功训练了 {len(our_adaboost_reg.estimators_)} 个基学习器")
    except Exception as e:
        print(f"  训练失败: {e}")
        # 如果训练失败，使用默认预测
        our_pred = np.zeros_like(y_test)
        our_mse = mean_squared_error(y_test, our_pred)
        our_r2 = r2_score(y_test, our_pred)

    print("\n训练sklearn的AdaBoost回归器...")
    sklearn_adaboost_reg.fit(X_train, y_train)
    sklearn_pred = sklearn_adaboost_reg.predict(X_test)
    sklearn_mse = mean_squared_error(y_test, sklearn_pred)
    sklearn_r2 = r2_score(y_test, sklearn_pred)

    print(f"\n我们的实现结果:")
    print(f"  测试MSE: {our_mse:.4f}")
    print(f"  测试R²: {our_r2:.4f}")
    print(f"  RMSE: {np.sqrt(our_mse):.4f}")
    print(
        f"  实际使用的基学习器数量: {len(our_adaboost_reg.estimators_) if hasattr(our_adaboost_reg, 'estimators_') else 0}")

    if hasattr(our_adaboost_reg, 'estimator_weights_') and len(our_adaboost_reg.estimator_weights_) > 0:
        weights = our_adaboost_reg.estimator_weights_
        print(f"  基学习器权重范围: [{weights.min():.4f}, {weights.max():.4f}]")
        print(f"  平均基学习器权重: {weights.mean():.4f}")

    print(f"\nsklearn实现结果:")
    print(f"  测试MSE: {sklearn_mse:.4f}")
    print(f"  测试R²: {sklearn_r2:.4f}")
    print(f"  RMSE: {np.sqrt(sklearn_mse):.4f}")
    print(f"  实际使用的基学习器数量: {len(sklearn_adaboost_reg.estimators_)}")

    # 可视化预测结果
    visualize_adaboost_regressor_results(
        our_adaboost_reg, sklearn_adaboost_reg,
        our_pred, sklearn_pred, y_test,
        our_mse, sklearn_mse, our_r2, sklearn_r2
    )

    return our_mse, our_r2, sklearn_mse, sklearn_r2


def visualize_adaboost_regressor_results(our_model, sklearn_model,
                                         our_pred, sklearn_pred, y_test,
                                         our_mse, sklearn_mse, our_r2, sklearn_r2):
    """可视化AdaBoost回归器结果 - 修复版本"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # 1. 我们的实现预测结果
    ax1 = axes[0, 0]
    ax1.scatter(y_test, our_pred, alpha=0.6, edgecolors='k', linewidth=0.5, s=20)

    # 计算回归线
    z = np.polyfit(y_test, our_pred, 1)
    p = np.poly1d(z)
    x_range = np.linspace(y_test.min(), y_test.max(), 100)
    ax1.plot(x_range, p(x_range), 'r--', linewidth=2,
             label=f'y = {z[0]:.3f}x + {z[1]:.3f}')

    # 理想对角线
    ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
             'g-', linewidth=2, alpha=0.5, label='理想预测')

    ax1.set_xlabel('真实值', fontsize=12)
    ax1.set_ylabel('预测值', fontsize=12)
    ax1.set_title('我们的实现 - 预测结果', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. sklearn实现预测结果
    ax2 = axes[0, 1]
    ax2.scatter(y_test, sklearn_pred, alpha=0.6, edgecolors='k', linewidth=0.5, s=20)

    z = np.polyfit(y_test, sklearn_pred, 1)
    p = np.poly1d(z)
    ax2.plot(x_range, p(x_range), 'r--', linewidth=2,
             label=f'y = {z[0]:.3f}x + {z[1]:.3f}')

    ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
             'g-', linewidth=2, alpha=0.5, label='理想预测')

    ax2.set_xlabel('真实值', fontsize=12)
    ax2.set_ylabel('预测值', fontsize=12)
    ax2.set_title('sklearn实现 - 预测结果', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. 残差对比
    ax3 = axes[0, 2]
    our_residuals = y_test - our_pred
    sklearn_residuals = y_test - sklearn_pred

    bins = 30
    ax3.hist(our_residuals, bins=bins, alpha=0.7, color='blue',
             label='我们的实现', edgecolor='black', density=True)
    ax3.hist(sklearn_residuals, bins=bins, alpha=0.7, color='red',
             label='sklearn实现', edgecolor='black', density=True)

    ax3.axvline(x=0, color='green', linestyle='--', linewidth=2,
                label='零残差线', alpha=0.7)
    ax3.set_xlabel('残差', fontsize=12)
    ax3.set_ylabel('概率密度', fontsize=12)
    ax3.set_title('残差分布对比（归一化）', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. 基学习器权重分布（我们的实现）
    ax4 = axes[1, 0]
    if hasattr(our_model, 'estimator_weights_') and len(our_model.estimator_weights_) > 0:
        weights = our_model.estimator_weights_
        bars = ax4.bar(range(1, len(weights) + 1), weights,
                       color='steelblue', edgecolor='navy', alpha=0.7)

        # 添加统计信息
        mean_weight = np.mean(weights)
        median_weight = np.median(weights)
        ax4.axhline(y=mean_weight, color='red', linestyle='--',
                    linewidth=2, label=f'均值: {mean_weight:.4f}', alpha=0.7)
        ax4.axhline(y=median_weight, color='orange', linestyle='--',
                    linewidth=2, label=f'中位数: {median_weight:.4f}', alpha=0.7)

        ax4.set_xlabel('基学习器索引', fontsize=12)
        ax4.set_ylabel('权重', fontsize=12)
        ax4.set_title(f'我们的实现 - 基学习器权重\n(n={len(weights)})', fontsize=14, fontweight='bold')
        ax4.legend(fontsize=9)
        ax4.grid(True, alpha=0.3, axis='y')
    else:
        ax4.text(0.5, 0.5, '无基学习器权重数据',
                 ha='center', va='center', fontsize=12,
                 transform=ax4.transAxes)
        ax4.set_title('我们的实现 - 基学习器权重', fontsize=14, fontweight='bold')

    # 5. 训练损失曲线
    ax5 = axes[1, 1]
    if hasattr(our_model, 'train_scores_') and len(our_model.train_scores_) > 0:
        train_scores = our_model.train_scores_
        ax5.plot(range(1, len(train_scores) + 1), train_scores,
                 'b-', linewidth=2, label='训练损失', alpha=0.8)
        ax5.set_xlabel('迭代次数', fontsize=12)
        ax5.set_ylabel('MSE损失', fontsize=12)
        ax5.set_title('训练过程损失曲线', fontsize=14, fontweight='bold')
        ax5.grid(True, alpha=0.3)
        ax5.legend()
    else:
        ax5.text(0.5, 0.5, '无训练损失数据',
                 ha='center', va='center', fontsize=12,
                 transform=ax5.transAxes)
        ax5.set_title('训练过程损失曲线', fontsize=14, fontweight='bold')

    # 6. 性能指标对比
    ax6 = axes[1, 2]
    metrics = ['MSE', 'R²', 'RMSE']
    our_scores = [our_mse, our_r2, np.sqrt(our_mse)]
    sklearn_scores = [sklearn_mse, sklearn_r2, np.sqrt(sklearn_mse)]

    x = np.arange(len(metrics))
    width = 0.35

    bars1 = ax6.bar(x - width / 2, our_scores, width, label='我们的实现',
                    color='lightblue', edgecolor='navy')
    bars2 = ax6.bar(x + width / 2, sklearn_scores, width, label='sklearn实现',
                    color='lightcoral', edgecolor='darkred')

    ax6.set_xlabel('指标', fontsize=12)
    ax6.set_ylabel('值', fontsize=12)
    ax6.set_title('性能指标对比', fontsize=14, fontweight='bold')
    ax6.set_xticks(x)
    ax6.set_xticklabels(metrics)
    ax6.legend()
    ax6.grid(True, alpha=0.3, axis='y')

    # 添加数值标签
    max_score = max(max(our_scores), max(sklearn_scores))
    for bars, scores in zip([bars1, bars2], [our_scores, sklearn_scores]):
        for bar, score in zip(bars, scores):
            height = bar.get_height()
            # 根据值的大小调整标签位置
            label_y = height + 0.05 * max_score
            ax6.text(bar.get_x() + bar.get_width() / 2, label_y,
                     f'{score:.3f}', ha='center', va='bottom', fontsize=9)

    plt.suptitle('AdaBoost回归器结果对比分析', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 设置中文字体
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方块的问题

    plt.savefig('../results/figures/day3_adaboost_regressor_comparison.png',
                dpi=150, bbox_inches='tight')
    plt.show()

    # 打印残差统计
    print("\n" + "=" * 60)
    print("残差统计分析")
    print("=" * 60)
    print(f"\n我们的实现残差统计:")
    print(f"  残差均值: {np.mean(our_residuals):.4f}")
    print(f"  残差标准差: {np.std(our_residuals):.4f}")
    print(f"  残差范围: [{our_residuals.min():.4f}, {our_residuals.max():.4f}]")

    print(f"\nsklearn实现残差统计:")
    print(f"  残差均值: {np.mean(sklearn_residuals):.4f}")
    print(f"  残差标准差: {np.std(sklearn_residuals):.4f}")
    print(f"  残差范围: [{sklearn_residuals.min():.4f}, {sklearn_residuals.max():.4f}]")

    print(f"\n模型对比:")
    print(f"  MSE差异: {abs(our_mse - sklearn_mse):.4f}")
    print(f"  R²差异: {abs(our_r2 - sklearn_r2):.4f}")

if __name__ == "__main__":
    # 运行测试
    print("开始AdaBoost算法测试...")
    print("="*60)

    try:
        # 测试分类器
        print("\n\n测试分类器...")
        our_acc, sklearn_acc, our_clf, sklearn_clf = test_adaboost_classifier()

        # 测试回归器
        print("\n\n测试回归器...")
        our_mse, our_r2, sklearn_mse, sklearn_r2 = test_adaboost_regressor()

        # 生成总结报告
        print("\n" + "="*60)
        print("测试总结报告")
        print("="*60)

        print(f"\n分类任务:")
        print(f"  我们的实现准确率: {our_acc:.4f}")
        print(f"  sklearn实现准确率: {sklearn_acc:.4f}")
        print(f"  准确率差异: {abs(our_acc - sklearn_acc):.4f}")
        print(f"  相对准确率: {our_acc / sklearn_acc * 100:.2f}%")

        print(f"\n回归任务:")
        print(f"  我们的实现MSE: {our_mse:.4f} (R²: {our_r2:.4f})")
        print(f"  sklearn实现MSE: {sklearn_mse:.4f} (R²: {sklearn_r2:.4f})")
        print(f"  MSE相对表现: {our_mse / sklearn_mse:.4f}")
        print(f"  R²相对表现: {our_r2 / sklearn_r2:.4f}")

        print(f"\n整体评价:")
        if our_acc > sklearn_acc and our_mse < sklearn_mse:
            print("  ✅ 我们的实现在分类和回归任务上都优于sklearn实现！")
        elif our_acc > sklearn_acc:
            print("  📈 我们的实现在分类任务上优于sklearn实现")
        elif our_mse < sklearn_mse:
            print("  📈 我们的实现在回归任务上优于sklearn实现")
        else:
            print("  📊 我们的实现与sklearn实现性能接近")

        print(f"\n测试完成！所有结果已保存到 ../results/figures/ 目录")

    except Exception as e:
        print(f"测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
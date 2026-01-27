"""Bagging和随机森林综合实验 - experiments/day2_bagging_experiments.py"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer, load_wine, load_digits
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier as SklearnBagging
from sklearn.ensemble import RandomForestClassifier as SklearnRF
from sklearn.ensemble import ExtraTreesClassifier
import warnings

warnings.filterwarnings('ignore')

# 导入我们的实现
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.bagging import BaggingClassifier
from models.random_forest import RandomForestClassifier


def compare_ensemble_methods(dataset_name='breast_cancer'):
    """对比不同集成方法"""
    print("\n" + "=" * 60)
    print(f"数据集: {dataset_name}")
    print("=" * 60)

    # 加载数据集
    if dataset_name == 'breast_cancer':
        data = load_breast_cancer()
        dataset_info = "乳腺癌数据集（二分类）"
    elif dataset_name == 'wine':
        data = load_wine()
        dataset_info = "红酒数据集（多分类）"
    elif dataset_name == 'digits':
        data = load_digits()
        dataset_info = "手写数字数据集（多分类）"
    else:
        raise ValueError(f"未知数据集: {dataset_name}")

    X, y = data.data, data.target

    print(f"数据集信息: {dataset_info}")
    print(f"样本数: {X.shape[0]}, 特征数: {X.shape[1]}, 类别数: {len(np.unique(y))}")

    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 基学习器
    base_tree = DecisionTreeClassifier(max_depth=5, random_state=42)

    # 定义所有模型
    models = {
        '决策树': DecisionTreeClassifier(max_depth=5, random_state=42),
        'Bagging (我们的实现)': BaggingClassifier(
            base_estimator=DecisionTreeClassifier(max_depth=5, random_state=42),
            n_estimators=50,
            max_samples=0.8,
            max_features=0.8,
            oob_score=True,
            random_state=42,
            verbose=0
        ),
        'Bagging (sklearn)': SklearnBagging(
            estimator=DecisionTreeClassifier(max_depth=5, random_state=42),
            n_estimators=50,
            max_samples=0.8,
            max_features=0.8,
            oob_score=True,
            random_state=42
        ),
        '随机森林 (我们的实现)': RandomForestClassifier(
            n_estimators=50,
            max_depth=5,
            max_features='sqrt',
            oob_score=True,
            random_state=42,
            verbose=0
        ),
        '随机森林 (sklearn)': SklearnRF(
            n_estimators=50,
            max_depth=5,
            max_features='sqrt',
            oob_score=True,
            random_state=42,
            n_jobs=-1
        ),
        '极端随机树 (sklearn)': ExtraTreesClassifier(
            n_estimators=50,
            max_depth=5,
            max_features='sqrt',
            random_state=42
        )
    }

    # 训练和评估
    results = {}

    for name, model in models.items():
        print(f"\n训练 {name}...")

        # 训练
        import time
        start_time = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start_time

        # 评估
        train_acc = model.score(X_train, y_train)
        test_acc = model.score(X_test, y_test)

        # 获取OOB分数（如果可用）
        oob_score = getattr(model, 'oob_score_', None)

        # 交叉验证
        cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy', n_jobs=-1)

        results[name] = {
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'oob_score': oob_score,
            'train_time': train_time
        }

        print(f"  训练时间: {train_time:.3f}s")
        print(f"  训练准确率: {train_acc:.4f}")
        print(f"  测试准确率: {test_acc:.4f}")
        print(f"  交叉验证: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        if oob_score is not None:
            print(f"  OOB分数: {oob_score:.4f}")

    return results, models, dataset_info


def visualize_comparison(results, dataset_name, dataset_info):
    """可视化对比结果"""
    names = list(results.keys())
    test_accs = [results[name]['test_accuracy'] for name in names]
    cv_means = [results[name]['cv_mean'] for name in names]
    cv_stds = [results[name]['cv_std'] for name in names]
    train_times = [results[name]['train_time'] for name in names]

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle(f'{dataset_info} - 不同集成方法对比', fontsize=16)

    x = np.arange(len(names))
    width = 0.35

    # 1. 测试准确率
    bars1 = axes[0, 0].bar(x, test_accs, width, color='skyblue', edgecolor='black')
    axes[0, 0].set_xlabel('模型')
    axes[0, 0].set_ylabel('测试准确率')
    axes[0, 0].set_title('测试准确率对比')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(names, rotation=45, ha='right')
    axes[0, 0].grid(True, alpha=0.3, axis='y')

    # 添加数值标签
    for bar in bars1:
        height = bar.get_height()
        axes[0, 0].annotate(f'{height:.3f}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=9)

    # 2. 交叉验证结果
    bars2 = axes[0, 1].bar(x, cv_means, yerr=cv_stds, capsize=5,
                           color='lightcoral', edgecolor='black', alpha=0.7)
    axes[0, 1].set_xlabel('模型')
    axes[0, 1].set_ylabel('准确率')
    axes[0, 1].set_title('5折交叉验证结果（均值±标准差）')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(names, rotation=45, ha='right')
    axes[0, 1].grid(True, alpha=0.3, axis='y')

    # 添加数值标签
    for bar, std in zip(bars2, cv_stds):
        height = bar.get_height()
        axes[0, 1].annotate(f'{height:.3f}±{std:.3f}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=8)

    # 3. OOB分数（如果有）
    oob_scores = []
    for name in names:
        oob = results[name]['oob_score']
        oob_scores.append(oob if oob is not None else 0)

    bars3 = axes[1, 0].bar(x, oob_scores, width, color='lightgreen', edgecolor='black')
    axes[1, 0].set_xlabel('模型')
    axes[1, 0].set_ylabel('OOB分数')
    axes[1, 0].set_title('OOB分数对比（支持OOB的模型）')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(names, rotation=45, ha='right')
    axes[1, 0].grid(True, alpha=0.3, axis='y')

    # 添加数值标签
    for bar, score in zip(bars3, oob_scores):
        if score > 0:
            height = bar.get_height()
            axes[1, 0].annotate(f'{height:.3f}',
                                xy=(bar.get_x() + bar.get_width() / 2, height),
                                xytext=(0, 3),
                                textcoords="offset points",
                                ha='center', va='bottom', fontsize=9)

    # 4. 训练时间
    bars4 = axes[1, 1].bar(x, train_times, width, color='gold', edgecolor='black')
    axes[1, 1].set_xlabel('模型')
    axes[1, 1].set_ylabel('训练时间（秒）')
    axes[1, 1].set_title('训练时间对比')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(names, rotation=45, ha='right')
    axes[1, 1].grid(True, alpha=0.3, axis='y')

    # 添加数值标签
    for bar in bars4:
        height = bar.get_height()
        axes[1, 1].annotate(f'{height:.3f}s',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=9)

    # 设置字体为系统自带的中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'KaiTi', 'FangSong']  # 设置中文字体
    # 解决负号显示问题
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方块的问题
    plt.tight_layout()
    plt.savefig(f'../results/figures/day2_comparison_{dataset_name}.png', dpi=150, bbox_inches='tight')
    plt.show()

    # 计算相对于决策树的提升
    baseline = results['决策树']['test_accuracy']
    print(f"\n相对于决策树的提升:")
    print("-" * 50)
    for name in names:
        if name != '决策树':
            improvement = (results[name]['test_accuracy'] - baseline) / baseline * 100
            print(f"{name:25s}: {improvement:6.2f}% (测试准确率: {results[name]['test_accuracy']:.4f})")


def run_all_datasets():
    """在所有数据集上运行实验"""
    datasets = ['breast_cancer', 'wine', 'digits']
    all_results = {}

    for dataset in datasets:
        results, models, dataset_info = compare_ensemble_methods(dataset)
        all_results[dataset] = results

        visualize_comparison(results, dataset, dataset_info)

    return all_results


if __name__ == "__main__":
    import os

    os.makedirs('../results/figures', exist_ok=True)

    print("开始综合对比实验...")
    all_results = run_all_datasets()

    # 保存结果
    import json


    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return super(NumpyEncoder, self).default(obj)


    with open('../results/logs/day2_comparison_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, cls=NumpyEncoder)

    print("\n" + "=" * 60)
    print("所有实验完成！结果已保存到 ../results/day2_comparison_results.json")
    print("=" * 60)
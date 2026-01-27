"""OOB深度分析 - experiments/oob_deep_analysis.py"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, load_breast_cancer, load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
import warnings

warnings.filterwarnings('ignore')


def oob_error_comprehensive_analysis():
    """OOB误差全面分析"""
    print("=" * 60)
    print("OOB误差全面分析")
    print("=" * 60)

    # 创建图形
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('OOB误差深度分析', fontsize=16, fontweight='bold')

    # 分析1: OOB误差与测试误差的对比（不同树数量）
    print("\n分析1: OOB误差与测试误差的对比")
    X, y = make_classification(
        n_samples=1000, n_features=20, n_informative=15,
        n_classes=2, random_state=42
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    n_estimators_range = list(range(1, 201, 10))
    oob_errors = []
    test_errors = []

    for n in n_estimators_range:
        rf = RandomForestClassifier(
            n_estimators=n,
            max_depth=5,
            oob_score=True,
            random_state=42,
            n_jobs=-1
        )
        rf.fit(X_train, y_train)

        oob_error = 1 - rf.oob_score_
        oob_errors.append(oob_error)

        test_error = 1 - rf.score(X_test, y_test)
        test_errors.append(test_error)

    # 子图1: OOB误差 vs 测试误差
    ax1 = axes[0, 0]
    ax1.plot(n_estimators_range, oob_errors, 'b-', label='OOB误差', linewidth=2, alpha=0.8)
    ax1.plot(n_estimators_range, test_errors, 'r-', label='测试误差', linewidth=2, alpha=0.8)
    ax1.fill_between(n_estimators_range, oob_errors, test_errors, alpha=0.2, color='gray')
    ax1.set_xlabel('树的数量')
    ax1.set_ylabel('误差')
    ax1.set_title('OOB误差 vs 测试误差')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 计算相关性
    correlation = np.corrcoef(oob_errors, test_errors)[0, 1]
    ax1.text(0.05, 0.95, f'相关性: {correlation:.3f}',
             transform=ax1.transAxes, fontsize=10,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # 子图2: 误差差异分布
    ax2 = axes[0, 1]
    error_differences = np.array(oob_errors) - np.array(test_errors)
    ax2.hist(error_differences, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
    ax2.axvline(x=np.mean(error_differences), color='red', linestyle='--',
                label=f'均值: {np.mean(error_differences):.4f}')
    ax2.axvline(x=0, color='green', linestyle='-', alpha=0.5)
    ax2.set_xlabel('OOB误差 - 测试误差')
    ax2.set_ylabel('频数')
    ax2.set_title('误差差异分布')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 分析2: OOB误差的统计特性
    print("\n分析2: OOB误差的统计特性")

    # 多次实验看OOB误差的稳定性
    n_repeats = 20
    oob_stability = []
    test_stability = []

    for _ in range(n_repeats):
        X, y = make_classification(
            n_samples=500, n_features=10, n_informative=8,
            n_classes=2, random_state=np.random.randint(1000)
        )

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        rf = RandomForestClassifier(
            n_estimators=50,
            oob_score=True,
            random_state=42
        )
        rf.fit(X_train, y_train)

        oob_stability.append(1 - rf.oob_score_)
        test_stability.append(1 - rf.score(X_test, y_test))

    # 子图3: OOB误差的稳定性
    ax3 = axes[0, 2]
    x_pos = range(n_repeats)
    width = 0.35

    ax3.bar([x - width / 2 for x in x_pos], oob_stability, width,
            label='OOB误差', color='blue', alpha=0.7)
    ax3.bar([x + width / 2 for x in x_pos], test_stability, width,
            label='测试误差', color='red', alpha=0.7)
    ax3.set_xlabel('实验序号')
    ax3.set_ylabel('误差')
    ax3.set_title('OOB误差稳定性分析')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')

    # 分析3: OOB作为早期停止标准
    print("\n分析3: OOB作为早期停止标准")

    n_estimators_large = list(range(1, 501, 20))
    oob_errors_large = []
    test_errors_large = []

    for n in n_estimators_large:
        rf = RandomForestClassifier(
            n_estimators=n,
            oob_score=True,
            random_state=42,
            n_jobs=-1
        )
        rf.fit(X_train, y_train)

        oob_errors_large.append(1 - rf.oob_score_)
        test_errors_large.append(1 - rf.score(X_test, y_test))

    # 子图4: OOB早期停止分析
    ax4 = axes[1, 0]
    ax4.plot(n_estimators_large, oob_errors_large, 'b-', label='OOB误差', linewidth=2)
    ax4.plot(n_estimators_large, test_errors_large, 'r-', label='测试误差', linewidth=2)

    # 找到OOB最小值
    min_oob_idx = np.argmin(oob_errors_large)
    min_oob_n = n_estimators_large[min_oob_idx]
    min_oob_error = oob_errors_large[min_oob_idx]

    # 找到测试误差最小值
    min_test_idx = np.argmin(test_errors_large)
    min_test_n = n_estimators_large[min_test_idx]
    min_test_error = test_errors_large[min_test_idx]

    ax4.axvline(x=min_oob_n, color='blue', linestyle='--', alpha=0.7,
                label=f'OOB最优: {min_oob_n}树')
    ax4.axvline(x=min_test_n, color='red', linestyle='--', alpha=0.7,
                label=f'测试最优: {min_test_n}树')

    ax4.set_xlabel('树的数量')
    ax4.set_ylabel('误差')
    ax4.set_title('OOB作为早期停止标准')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # 分析4: 不同数据集上的OOB可靠性
    print("\n分析4: 不同数据集上的OOB可靠性")

    datasets = {
        '模拟数据': make_classification(n_samples=1000, n_features=20,
                                        n_informative=15, random_state=42),
        '乳腺癌': load_breast_cancer(return_X_y=True),
        '手写数字': load_digits(return_X_y=True)
    }

    dataset_names = []
    oob_test_correlations = []
    oob_test_mae = []  # 平均绝对误差

    for name, (X, y) in datasets.items():
        if name == '手写数字':
            # 手写数字数据集较大，采样以加快计算
            idx = np.random.choice(len(X), 1000, replace=False)
            X, y = X[idx], y[idx]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        n_trees_range = [10, 20, 30, 50, 100]
        oob_errors_ds = []
        test_errors_ds = []

        for n in n_trees_range:
            rf = RandomForestClassifier(
                n_estimators=n,
                oob_score=True,
                random_state=42,
                n_jobs=-1
            )
            rf.fit(X_train, y_train)

            oob_errors_ds.append(1 - rf.oob_score_)
            test_errors_ds.append(1 - rf.score(X_test, y_test))

        # 计算相关性
        if len(oob_errors_ds) > 1:
            corr = np.corrcoef(oob_errors_ds, test_errors_ds)[0, 1]
            mae = np.mean(np.abs(np.array(oob_errors_ds) - np.array(test_errors_ds)))
        else:
            corr = 0
            mae = 0

        dataset_names.append(name)
        oob_test_correlations.append(corr)
        oob_test_mae.append(mae)

    # 子图5: 不同数据集的OOB可靠性
    ax5 = axes[1, 1]
    x_pos = range(len(dataset_names))
    bars1 = ax5.bar(x_pos, oob_test_correlations, color='lightgreen',
                    edgecolor='black', alpha=0.7)

    ax5.set_xlabel('数据集')
    ax5.set_ylabel('相关性')
    ax5.set_title('不同数据集上OOB与测试误差的相关性')
    ax5.set_xticks(x_pos)
    ax5.set_xticklabels(dataset_names, rotation=45)
    ax5.axhline(y=0.9, color='red', linestyle='--', alpha=0.5, label='高可靠性阈值')
    ax5.legend()
    ax5.grid(True, alpha=0.3, axis='y')

    # 添加数值标签
    for bar, corr in zip(bars1, oob_test_correlations):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                 f'{corr:.3f}', ha='center', va='bottom', fontsize=9)

    # 分析5: OOB误差置信区间
    print("\n分析5: OOB误差置信区间分析")

    # 通过Bootstrap计算OOB误差的置信区间
    n_bootstraps = 1000
    oob_bootstrap_errors = []

    X, y = make_classification(n_samples=500, n_features=15,
                               n_informative=10, random_state=42)

    for _ in range(n_bootstraps):
        # Bootstrap采样
        indices = np.random.choice(len(X), len(X), replace=True)
        X_boot, y_boot = X[indices], y[indices]

        # 训练随机森林
        rf = RandomForestClassifier(
            n_estimators=50,
            oob_score=True,
            random_state=42
        )
        rf.fit(X_boot, y_boot)

        oob_bootstrap_errors.append(1 - rf.oob_score_)

    # 计算置信区间
    oob_mean = np.mean(oob_bootstrap_errors)
    oob_std = np.std(oob_bootstrap_errors)
    ci_lower = np.percentile(oob_bootstrap_errors, 2.5)
    ci_upper = np.percentile(oob_bootstrap_errors, 97.5)

    # 子图6: OOB误差置信区间
    ax6 = axes[1, 2]

    # 绘制直方图
    ax6.hist(oob_bootstrap_errors, bins=30, color='lightblue',
             edgecolor='black', alpha=0.7, density=True)

    # 添加正态分布曲线
    from scipy.stats import norm
    x = np.linspace(ci_lower - 0.1, ci_upper + 0.1, 1000)
    ax6.plot(x, norm.pdf(x, oob_mean, oob_std), 'r-',
             label=f'正态分布\nμ={oob_mean:.3f}, σ={oob_std:.3f}')

    # 添加置信区间
    ax6.axvline(x=ci_lower, color='green', linestyle='--',
                label=f'95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]')
    ax6.axvline(x=ci_upper, color='green', linestyle='--')
    ax6.axvline(x=oob_mean, color='blue', linestyle='-',
                label=f'均值: {oob_mean:.3f}')

    ax6.set_xlabel('OOB误差')
    ax6.set_ylabel('密度')
    ax6.set_title('OOB误差Bootstrap置信区间')
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    # 设置字体为系统自带的中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'KaiTi', 'FangSong']  # 设置中文字体
    # 解决负号显示问题
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方块的问题
    plt.tight_layout()
    plt.savefig('../results/figures/day2_oob_deep_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()

    # 输出详细分析结果
    print("\n" + "=" * 60)
    print("OOB深度分析总结")
    print("=" * 60)

    print(f"\n1. OOB与测试误差的相关性: {correlation:.4f}")
    print(f"   平均绝对差异: {np.mean(np.abs(error_differences)):.4f}")

    print(f"\n2. OOB作为早期停止标准:")
    print(f"   OOB推荐树数量: {min_oob_n} (误差: {min_oob_error:.4f})")
    print(f"   测试最优树数量: {min_test_n} (误差: {min_test_error:.4f})")
    print(f"   差异: {abs(min_oob_n - min_test_n)} 棵树")

    print(f"\n3. 不同数据集的OOB可靠性:")
    for name, corr, mae in zip(dataset_names, oob_test_correlations, oob_test_mae):
        reliability = "高 ✓" if corr > 0.9 else "中" if corr > 0.7 else "低 ⚠"
        print(f"   {name:10s}: 相关性={corr:.3f}, MAE={mae:.4f}, 可靠性={reliability}")

    print(f"\n4. OOB误差置信区间:")
    print(f"   均值: {oob_mean:.4f}")
    print(f"   标准差: {oob_std:.4f}")
    print(f"   95%置信区间: [{ci_lower:.4f}, {ci_upper:.4f}]")
    print(f"   区间宽度: {ci_upper - ci_lower:.4f}")

    print(f"\n5. 实用建议:")
    print(f"   - OOB与测试误差平均差异: {np.mean(np.abs(error_differences)):.3f}")
    print(f"   - 当差异 < 0.02 时，OOB可作为可靠估计")
    print(f"   - 树数量 ≥ 20 时，OOB稳定性较高")
    print(f"   - 对于关键应用，建议用交叉验证验证OOB估计")

    return {
        'correlation': correlation,
        'error_differences_mean': np.mean(np.abs(error_differences)),
        'oob_stability_mean': np.mean(oob_stability),
        'oob_stability_std': np.std(oob_stability),
        'optimal_n_oob': min_oob_n,
        'optimal_n_test': min_test_n,
        'dataset_correlations': dict(zip(dataset_names, oob_test_correlations)),
        'bootstrap_ci': [ci_lower, ci_upper, ci_upper - ci_lower]
    }


def oob_statistical_properties():
    """分析OOB误差的统计特性"""
    print("\n" + "=" * 60)
    print("OOB误差统计特性分析")
    print("=" * 60)

    # 生成较大数据集进行统计特性分析
    X, y = make_classification(
        n_samples=2000, n_features=20, n_informative=15,
        n_classes=2, random_state=42
    )

    # 分析不同树数量下的OOB统计特性
    n_estimators_options = [10, 20, 50, 100, 200]

    print(f"{'树数量':<10} {'OOB均值':<10} {'OOB标准差':<10} {'偏度':<10} {'峰度':<10} {'有效样本比例':<15}")
    print("-" * 70)

    results = {}

    for n in n_estimators_options:
        rf = RandomForestClassifier(
            n_estimators=n,
            oob_score=True,
            random_state=42,
            n_jobs=-1
        )
        rf.fit(X, y)

        # 计算OOB样本比例
        n_samples = X.shape[0]
        oob_count = np.zeros(n_samples)

        # 简化的OOB样本统计
        # 注意：这里我们通过模拟来计算OOB样本比例
        # 实际上，每个样本是OOB的概率 = (1-1/n)^n ≈ 0.368

        # 计算理论OOB比例
        theoretical_oob_ratio = (1 - 1 / n) ** n

        # 从随机森林获取OOB决策函数
        if hasattr(rf, 'oob_decision_function_'):
            oob_decision = rf.oob_decision_function_
            actual_oob_ratio = np.sum(~np.isnan(oob_decision[:, 0])) / n_samples

            # 计算OOB预测的统计特性
            oob_predictions = np.argmax(oob_decision, axis=1)
            oob_correct = oob_predictions == y[~np.isnan(oob_decision[:, 0])]
            oob_error = 1 - np.mean(oob_correct)

            # 计算偏度和峰度
            from scipy.stats import skew, kurtosis
            oob_errors_per_sample = []

            # 简化：计算每个OOB样本的预测置信度
            for i in range(n_samples):
                if not np.isnan(oob_decision[i, 0]):
                    true_class = y[i]
                    confidence = oob_decision[i, true_class]
                    oob_errors_per_sample.append(1 - confidence)

            if len(oob_errors_per_sample) > 0:
                skewness = skew(oob_errors_per_sample)
                kurt = kurtosis(oob_errors_per_sample)
            else:
                skewness = kurt = np.nan

            print(
                f"{n:<10} {oob_error:<10.4f} {np.std(oob_errors_per_sample) if len(oob_errors_per_sample) > 0 else np.nan:<10.4f} "
                f"{skewness:<10.4f} {kurt:<10.4f} {actual_oob_ratio:<15.4f}")

            results[n] = {
                'oob_error': oob_error,
                'std': np.std(oob_errors_per_sample) if len(oob_errors_per_sample) > 0 else np.nan,
                'skewness': skewness,
                'kurtosis': kurt,
                'actual_oob_ratio': actual_oob_ratio,
                'theoretical_oob_ratio': theoretical_oob_ratio
            }
        else:
            print(f"{n:<10} {'N/A':<10} {'N/A':<10} {'N/A':<10} {'N/A':<10} {'N/A':<15}")

    return results


if __name__ == "__main__":
    import os

    os.makedirs('../results/figures', exist_ok=True)

    try:
        # 运行OOB全面分析
        analysis_results = oob_error_comprehensive_analysis()

        # 运行OOB统计特性分析
        stat_results = oob_statistical_properties()

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


        all_results = {
            'comprehensive_analysis': analysis_results,
            'statistical_properties': stat_results
        }

        with open('../results/logs/day2_oob_analysis.json', 'w') as f:
            json.dump(all_results, f, indent=2, cls=NumpyEncoder)

        print("\n✅ OOB深度分析完成！")
        print("图表已保存到 ../results/figures/day2_oob_deep_analysis.png")
        print("详细结果已保存到 ../results/logs/day2_oob_analysis.json")

    except Exception as e:
        print(f"\n❌ OOB深度分析失败: {e}")
        import traceback

        traceback.print_exc()
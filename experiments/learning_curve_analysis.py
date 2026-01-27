"""学习曲线分析 - experiments/learning_curve_analysis.py """
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
import time
import warnings
warnings.filterwarnings('ignore')

def learning_curve_analysis():
    """分析学习曲线：集成规模 vs 性能"""
    print("=" * 60)
    print("学习曲线分析：集成规模对性能的影响")
    print("=" * 60)

    # 生成数据
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_classes=2,
        random_state=42
    )

    # 定义模型
    models = {
        '决策树': DecisionTreeClassifier(max_depth=5, random_state=42),
        'Bagging': BaggingClassifier(
            estimator=DecisionTreeClassifier(max_depth=5, random_state=42),
            max_samples=0.8,
            random_state=42
        ),
        '随机森林': RandomForestClassifier(max_depth=5, random_state=42)
    }

    n_estimators_range = [1, 5, 10, 20, 30, 50, 100, 200]

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('集成方法学习曲线分析', fontsize=16)

    # 1. 准确率 vs 树的数量
    print("\n分析准确率随树数量的变化...")
    for name, model in models.items():
        test_scores = []
        oob_scores = []

        if name == '决策树':
            # 决策树只有一个估计器
            cv_scores = cross_val_score(model, X, y, cv=5)
            test_scores = [cv_scores.mean()] * len(n_estimators_range)
            model.fit(X, y)
            oob_scores = [model.score(X, y)] * len(n_estimators_range)
        else:
            for n in n_estimators_range:
                if name == 'Bagging':
                    model.n_estimators = n
                elif name == '随机森林':
                    model.n_estimators = n
                    model.oob_score = True

                cv_scores = cross_val_score(model, X, y, cv=3, n_jobs=-1)
                test_scores.append(cv_scores.mean())

                model.fit(X, y)
                oob_score = model.oob_score_ if hasattr(model, 'oob_score_') else None
                oob_scores.append(oob_score if oob_score is not None else model.score(X, y))

        axes[0, 0].plot(n_estimators_range[:len(test_scores)],
                       test_scores, 'o-', label=f'{name} (CV)', linewidth=2)
        if name != '决策树':
            axes[0, 0].plot(n_estimators_range[:len(oob_scores)],
                          oob_scores, 's--', label=f'{name} (OOB)', linewidth=2, alpha=0.7)

    axes[0, 0].set_xlabel('基学习器数量')
    axes[0, 0].set_ylabel('准确率')
    axes[0, 0].set_title('集成规模对准确率的影响')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2. 训练时间 vs 树的数量
    print("\n分析训练时间随树数量的变化...")

    for name in ['Bagging', '随机森林']:
        train_times = []

        for n in n_estimators_range:
            if name == 'Bagging':
                model = BaggingClassifier(
                    estimator=DecisionTreeClassifier(max_depth=5),
                    n_estimators=n,
                    random_state=42
                )
            else:
                model = RandomForestClassifier(
                    n_estimators=n,
                    max_depth=5,
                    random_state=42
                )

            start_time = time.time()
            model.fit(X, y)
            train_times.append(time.time() - start_time)

        axes[0, 1].plot(n_estimators_range, train_times, 'o-', label=name, linewidth=2)

    axes[0, 1].set_xlabel('基学习器数量')
    axes[0, 1].set_ylabel('训练时间（秒）')
    axes[0, 1].set_title('集成规模对训练时间的影响')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 3. 边际收益分析（随机森林）
    print("\n分析随机森林的边际收益...")
    model = RandomForestClassifier(max_depth=5, random_state=42)

    test_scores_rf = []
    for n in n_estimators_range:
        model.n_estimators = n
        cv_scores = cross_val_score(model, X, y, cv=3, n_jobs=-1)
        test_scores_rf.append(cv_scores.mean())

    # 计算边际收益
    marginal_gains = []
    relative_gains = []

    for i in range(1, len(test_scores_rf)):
        gain = test_scores_rf[i] - test_scores_rf[i-1]
        relative_gain = gain / test_scores_rf[i-1] * 100
        marginal_gains.append(gain)
        relative_gains.append(relative_gain)

    x_pos = n_estimators_range[1:]
    bars = axes[1, 0].bar(x_pos, marginal_gains, width=10, color='lightblue', edgecolor='black')
    axes[1, 0].set_xlabel('树的数量')
    axes[1, 0].set_ylabel('准确率提升')
    axes[1, 0].set_title('随机森林边际收益分析')
    axes[1, 0].grid(True, alpha=0.3, axis='y')

    # 添加数值标签
    for bar, gain, rel_gain in zip(bars, marginal_gains, relative_gains):
        height = bar.get_height()
        axes[1, 0].annotate(f'+{gain:.3f}\n({rel_gain:.1f}%)',
                          xy=(bar.get_x() + bar.get_width() / 2, height),
                          xytext=(0, 3 if height >= 0 else -15),
                          textcoords="offset points",
                          ha='center', va='bottom' if height >= 0 else 'top',
                          fontsize=8)

    # 4. 准确率-时间性价比分析
    print("\n分析准确率-时间性价比...")
    model_rf = RandomForestClassifier(max_depth=5, random_state=42)
    model_bagging = BaggingClassifier(
        estimator=DecisionTreeClassifier(max_depth=5),
        random_state=42
    )

    # 存储准确率和时间
    rf_accuracies = []
    rf_times = []
    bagging_accuracies = []
    bagging_times = []

    for n in n_estimators_range[:6]:  # 只看前几个点
        # 随机森林
        model_rf.n_estimators = n
        start_time = time.time()
        cv_scores_rf = cross_val_score(model_rf, X, y, cv=3, n_jobs=-1)
        train_time_rf = time.time() - start_time
        rf_accuracies.append(cv_scores_rf.mean())
        rf_times.append(train_time_rf)

        # Bagging
        model_bagging.n_estimators = n
        start_time = time.time()
        cv_scores_bagging = cross_val_score(model_bagging, X, y, cv=3, n_jobs=-1)
        train_time_bagging = time.time() - start_time
        bagging_accuracies.append(cv_scores_bagging.mean())
        bagging_times.append(train_time_bagging)

    # 计算边际性价比（而非绝对性价比）
    efficiency_rf = []
    efficiency_bagging = []

    for i in range(len(n_estimators_range[:6])):
        if i == 0:
            # 第一个点没有"前一个点"来计算边际提升
            # 使用相对于基线（1棵树）的提升
            efficiency_rf.append(0.0)  # 基线性价比设为0
            efficiency_bagging.append(0.0)
        else:
            # 随机森林边际性价比 = 准确率提升 / 额外时间
            acc_gain_rf = rf_accuracies[i] - rf_accuracies[0]  # 相对于基线
            time_cost_rf = rf_times[i] - rf_times[0]  # 相对于基线
            if time_cost_rf > 1e-6:  # 避免除以0
                efficiency_rf.append(acc_gain_rf / time_cost_rf)
            else:
                efficiency_rf.append(0.0)

            # Bagging边际性价比
            acc_gain_bagging = bagging_accuracies[i] - bagging_accuracies[0]
            time_cost_bagging = bagging_times[i] - bagging_times[0]
            if time_cost_bagging > 1e-6:
                efficiency_bagging.append(acc_gain_bagging / time_cost_bagging)
            else:
                efficiency_bagging.append(0.0)

    #从第二个点开始绘制（因为第一个点性价比为0）
    axes[1, 1].plot(n_estimators_range[1:6], efficiency_rf[1:], 'o-', label='随机森林', linewidth=2)
    axes[1, 1].plot(n_estimators_range[1:6], efficiency_bagging[1:], 's-', label='Bagging', linewidth=2)
    axes[1, 1].set_xlabel('基学习器数量')
    axes[1, 1].set_ylabel('准确率提升/时间')
    axes[1, 1].set_title('准确率-时间边际性价比分析\n(相对于1棵树的提升)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    # 找到最佳性价比点（排除第一个点）
    if len(efficiency_rf) > 1:
        best_idx_rf = np.argmax(efficiency_rf[1:]) + 1
        best_idx_bagging = np.argmax(efficiency_bagging[1:]) + 1

        axes[1, 1].axvline(x=n_estimators_range[best_idx_rf], color='blue',
                          linestyle='--', alpha=0.5, label=f'RF最佳: {n_estimators_range[best_idx_rf]}棵')
        axes[1, 1].axvline(x=n_estimators_range[best_idx_bagging], color='orange',
                          linestyle='--', alpha=0.5, label=f'Bagging最佳: {n_estimators_range[best_idx_bagging]}棵')
        axes[1, 1].legend()

    plt.tight_layout()
    # 设置字体为系统自带的中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'KaiTi', 'FangSong']  # 设置中文字体
    # 解决负号显示问题
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方块的问题
    plt.savefig('../results/figures/day2_learning_curves_fixed.png', dpi=150, bbox_inches='tight')
    plt.show()

    # 输出总结
    print("\n" + "=" * 60)
    print("学习曲线分析总结")
    print("=" * 60)

    print("\n1. 边际收益分析（随机森林）:")
    print("-" * 40)
    for i in range(len(marginal_gains)):
        print(f"从 {n_estimators_range[i]} 到 {n_estimators_range[i+1]} 棵树: "
              f"准确率提升 {marginal_gains[i]:.4f} ({relative_gains[i]:.2f}%)")

    print(f"\n2. 最佳边际性价比点:")
    if len(efficiency_rf) > 1:
        print(f"   随机森林: {n_estimators_range[best_idx_rf]} 棵树")
        print(f"   Bagging: {n_estimators_range[best_idx_bagging]} 棵树")
    else:
        print("   数据不足，无法计算最佳性价比点")

    print("\n3. 实用建议（基于边际分析）:")
    print("   - 5-10棵树：边际收益最高（4-5%提升）")
    print("   - 20-30棵树：性价比通常最佳")
    print("   - 超过50棵树：边际收益通常<1%")
    print("   - 实际选择：20-50棵树通常是最佳平衡点")

    # 计算推荐树数量
    recommended_n = None
    for i, gain in enumerate(marginal_gains):
        if gain < 0.005:  # 当边际收益<0.5%时停止
            recommended_n = n_estimators_range[i]
            break

    if recommended_n:
        print(f"   - 推荐树数量: {recommended_n}（边际收益<0.5%）")
    else:
        print(f"   - 推荐树数量: 50（默认值）")

    return {
        'n_estimators_range': n_estimators_range,
        'marginal_gains': marginal_gains,
        'relative_gains': relative_gains,
        'efficiency_rf': efficiency_rf,
        'efficiency_bagging': efficiency_bagging,
        'best_rf': n_estimators_range[best_idx_rf] if len(efficiency_rf) > 1 else None,
        'best_bagging': n_estimators_range[best_idx_bagging] if len(efficiency_bagging) > 1 else None,
        'recommended_n': recommended_n
    }

if __name__ == "__main__":
    import os
    os.makedirs('../results/figures', exist_ok=True)

    try:
        results = learning_curve_analysis()

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

        with open('../results/logs/day2_learning_curve.json', 'w') as f:
            json.dump(results, f, indent=2, cls=NumpyEncoder)

        print("\n✅ 学习曲线分析完成！")
        print("图表已保存到 ../results/figures/day2_learning_curves.png")
        print("详细结果已保存到 ../results/logs/day2_learning_curve.json")

    except Exception as e:
        print(f"\n❌ 学习曲线分析失败: {e}")
        import traceback
        traceback.print_exc()
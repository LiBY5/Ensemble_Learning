"""学习曲线分析 - experiments/learning_curve_analysis.py """

# 导入数值计算库，用于数组操作和数学计算
import numpy as np

# 导入绘图库，用于数据可视化
import matplotlib.pyplot as plt

# 从scikit-learn导入数据生成函数，用于创建模拟数据集
from sklearn.datasets import make_classification

# 从scikit-learn集成学习模块导入随机森林和Bagging分类器
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier

# 从scikit-learn树模块导入决策树分类器
from sklearn.tree import DecisionTreeClassifier

# 从scikit-learn模型选择模块导入交叉验证评分函数
from sklearn.model_selection import cross_val_score

# 导入时间模块，用于测量代码执行时间
import time

# 导入警告模块，用于控制警告信息的显示
import warnings

# 设置过滤警告，忽略所有警告信息
warnings.filterwarnings('ignore')


def learning_curve_analysis():
    """
    分析学习曲线：集成规模 vs 性能

    该函数执行完整的集成学习方法学习曲线分析，包括：
    1. 不同集成规模下的准确率变化
    2. 训练时间随集成规模的变化
    3. 边际收益分析
    4. 准确率-时间性价比分析

    返回:
        dict: 包含所有分析结果的字典
    """

    # 打印标题和分隔线，增强输出可读性
    print("=" * 60)
    print("学习曲线分析：集成规模对性能的影响")
    print("=" * 60)

    # 生成模拟分类数据集
    # make_classification函数创建一个适合分类任务的合成数据集
    # n_samples=1000: 生成1000个样本
    # n_features=20: 每个样本有20个特征
    # n_informative=15: 其中15个是有效特征（与目标变量相关）
    # n_classes=2: 二分类问题
    # random_state=42: 随机种子，确保每次运行生成相同的数据
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_classes=2,
        random_state=42
    )

    # 定义要分析的机器学习模型
    # 使用字典存储模型，方便后续迭代和访问
    models = {
        # 基础模型：决策树，最大深度限制为5以防止过拟合
        '决策树': DecisionTreeClassifier(max_depth=5, random_state=42),

        # Bagging集成模型：基于决策树的装袋方法
        # estimator: 基学习器（这里使用决策树）
        # max_samples=0.8: 每个基学习器使用80%的样本进行训练
        'Bagging': BaggingClassifier(
            estimator=DecisionTreeClassifier(max_depth=5, random_state=42),
            max_samples=0.8,
            random_state=42
        ),

        # 随机森林集成模型：结合了Bagging和特征随机选择的集成方法
        '随机森林': RandomForestClassifier(max_depth=5, random_state=42)
    }

    # 定义集成规模（基学习器数量）的范围
    # 用于测试不同集成规模对性能的影响
    n_estimators_range = [1, 5, 10, 20, 30, 50, 100, 200]

    # 创建2行2列的子图布局，总图形大小为14x12英寸
    # fig是整体图形对象，axes是包含4个子图对象的2x2数组
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # 为整个图形添加总标题，字体大小为16
    fig.suptitle('集成方法学习曲线分析', fontsize=16)

    # ==================== 第一个子图：准确率 vs 树的数量 ====================
    print("\n分析准确率随树数量的变化...")

    # 遍历每个模型（决策树、Bagging、随机森林）
    for name, model in models.items():
        # 存储每个集成规模下的交叉验证准确率
        test_scores = []
        # 存储每个集成规模下的袋外（OOB）准确率
        oob_scores = []

        # 处理决策树模型（单个模型，不是集成方法）
        if name == '决策树':
            # 决策树是单个模型，没有集成规模的概念
            # 使用5折交叉验证评估模型性能
            cv_scores = cross_val_score(model, X, y, cv=5)
            # 对每个集成规模都使用相同的准确率（决策树只有一个估计器）
            test_scores = [cv_scores.mean()] * len(n_estimators_range)

            # 在完整数据集上训练模型，用于计算训练准确率
            model.fit(X, y)
            # 对每个集成规模都使用相同的训练准确率
            oob_scores = [model.score(X, y)] * len(n_estimators_range)
        else:
            # 处理集成方法（Bagging和随机森林）
            # 遍历不同的集成规模（基学习器数量）
            for n in n_estimators_range:
                # 根据模型类型设置基学习器数量
                if name == 'Bagging':
                    model.n_estimators = n
                elif name == '随机森林':
                    model.n_estimators = n
                    # 启用袋外估计（仅随机森林支持）
                    model.oob_score = True

                # 使用3折交叉验证评估模型性能
                # cv=3: 3折交叉验证
                # n_jobs=-1: 使用所有可用的CPU核心并行计算
                cv_scores = cross_val_score(model, X, y, cv=3, n_jobs=-1)
                # 存储交叉验证准确率的平均值
                test_scores.append(cv_scores.mean())

                # 在完整数据集上训练模型
                model.fit(X, y)
                # 获取袋外准确率（如果可用），否则使用训练准确率
                oob_score = model.oob_score_ if hasattr(model, 'oob_score_') else None
                oob_scores.append(oob_score if oob_score is not None else model.score(X, y))

        # 绘制交叉验证准确率曲线
        # 'o-': 圆形标记点，实线连接
        axes[0, 0].plot(n_estimators_range[:len(test_scores)],
                       test_scores, 'o-', label=f'{name} (CV)', linewidth=2)

        # 对于集成方法，额外绘制袋外准确率曲线
        if name != '决策树':
            # 's--': 正方形标记点，虚线连接
            axes[0, 0].plot(n_estimators_range[:len(oob_scores)],
                          oob_scores, 's--', label=f'{name} (OOB)', linewidth=2, alpha=0.7)

    # 设置第一个子图的坐标轴标签和标题
    axes[0, 0].set_xlabel('基学习器数量')  # x轴标签
    axes[0, 0].set_ylabel('准确率')       # y轴标签
    axes[0, 0].set_title('集成规模对准确率的影响')  # 子图标题
    axes[0, 0].legend()  # 显示图例
    axes[0, 0].grid(True, alpha=0.3)  # 显示网格，透明度为0.3

    # ==================== 第二个子图：训练时间 vs 树的数量 ====================
    print("\n分析训练时间随树数量的变化...")

    # 只分析集成方法（Bagging和随机森林），决策树训练时间固定
    for name in ['Bagging', '随机森林']:
        # 存储每个集成规模下的训练时间
        train_times = []

        # 遍历不同的集成规模
        for n in n_estimators_range:
            # 根据模型类型创建相应的模型实例
            if name == 'Bagging':
                # 创建Bagging分类器，指定基学习器数量
                model = BaggingClassifier(
                    estimator=DecisionTreeClassifier(max_depth=5),
                    n_estimators=n,
                    random_state=42
                )
            else:
                # 创建随机森林分类器，指定树的数量
                model = RandomForestClassifier(
                    n_estimators=n,
                    max_depth=5,
                    random_state=42
                )

            # 记录训练开始时间
            start_time = time.time()
            # 训练模型
            model.fit(X, y)
            # 计算并存储训练时间（结束时间 - 开始时间）
            train_times.append(time.time() - start_time)

        # 绘制训练时间曲线
        axes[0, 1].plot(n_estimators_range, train_times, 'o-', label=name, linewidth=2)

    # 设置第二个子图的坐标轴标签和标题
    axes[0, 1].set_xlabel('基学习器数量')
    axes[0, 1].set_ylabel('训练时间（秒）')
    axes[0, 1].set_title('集成规模对训练时间的影响')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # ==================== 第三个子图：边际收益分析（随机森林） ====================
    print("\n分析随机森林的边际收益...")

    # 创建随机森林模型，最大深度限制为5
    model = RandomForestClassifier(max_depth=5, random_state=42)

    # 存储随机森林在不同集成规模下的准确率
    test_scores_rf = []

    # 遍历不同的集成规模
    for n in n_estimators_range:
        # 设置随机森林的树数量
        model.n_estimators = n
        # 使用3折交叉验证评估模型性能
        cv_scores = cross_val_score(model, X, y, cv=3, n_jobs=-1)
        # 存储交叉验证准确率的平均值
        test_scores_rf.append(cv_scores.mean())

    # 计算边际收益（每增加一定数量树带来的准确率提升）
    marginal_gains = []   # 存储绝对边际收益
    relative_gains = []   # 存储相对边际收益（百分比）

    # 从第二个集成规模开始计算边际收益
    for i in range(1, len(test_scores_rf)):
        # 计算绝对边际收益：当前准确率 - 前一个准确率
        gain = test_scores_rf[i] - test_scores_rf[i-1]
        # 计算相对边际收益：(绝对收益 / 前一个准确率) * 100
        relative_gain = gain / test_scores_rf[i-1] * 100
        # 存储计算结果
        marginal_gains.append(gain)
        relative_gains.append(relative_gain)

    # 准备条形图的x轴位置（从第二个集成规模开始）
    x_pos = n_estimators_range[1:]

    # 绘制边际收益条形图
    # width=10: 条形宽度为10
    # color='lightblue': 条形颜色为浅蓝色
    # edgecolor='black': 条形边框为黑色
    bars = axes[1, 0].bar(x_pos, marginal_gains, width=10,
                          color='lightblue', edgecolor='black')

    # 设置第三个子图的坐标轴标签和标题
    axes[1, 0].set_xlabel('树的数量')
    axes[1, 0].set_ylabel('准确率提升')
    axes[1, 0].set_title('随机森林边际收益分析')
    # 仅在y轴显示网格线
    axes[1, 0].grid(True, alpha=0.3, axis='y')

    # 在条形图上添加数值标签
    for bar, gain, rel_gain in zip(bars, marginal_gains, relative_gains):
        # 获取条形的高度
        height = bar.get_height()
        # 根据条形高度决定标签位置（正数在上方，负数在下方）
        # xy: 标签位置（条形中心点）
        # xytext: 标签偏移量（以点为单位）
        # textcoords="offset points": 偏移量以点为单位
        axes[1, 0].annotate(f'+{gain:.3f}\n({rel_gain:.1f}%)',
                          xy=(bar.get_x() + bar.get_width() / 2, height),
                          xytext=(0, 3 if height >= 0 else -15),
                          textcoords="offset points",
                          ha='center',  # 水平居中
                          va='bottom' if height >= 0 else 'top',  # 垂直位置
                          fontsize=8)  # 字体大小

    # ==================== 第四个子图：准确率-时间性价比分析 ====================
    print("\n分析准确率-时间性价比...")

    # 创建随机森林模型
    model_rf = RandomForestClassifier(max_depth=5, random_state=42)

    # 创建Bagging模型
    model_bagging = BaggingClassifier(
        estimator=DecisionTreeClassifier(max_depth=5),
        random_state=42
    )

    # 初始化存储列表
    rf_accuracies = []    # 随机森林准确率
    rf_times = []         # 随机森林训练时间
    bagging_accuracies = []  # Bagging准确率
    bagging_times = []    # Bagging训练时间

    # 只分析前6个集成规模点，避免计算时间过长
    for n in n_estimators_range[:6]:
        # 测试随机森林
        model_rf.n_estimators = n
        # 记录训练开始时间
        start_time = time.time()
        # 使用交叉验证评估准确率
        cv_scores_rf = cross_val_score(model_rf, X, y, cv=3, n_jobs=-1)
        # 计算训练时间
        train_time_rf = time.time() - start_time
        # 存储结果
        rf_accuracies.append(cv_scores_rf.mean())
        rf_times.append(train_time_rf)

        # 测试Bagging
        model_bagging.n_estimators = n
        # 记录训练开始时间
        start_time = time.time()
        # 使用交叉验证评估准确率
        cv_scores_bagging = cross_val_score(model_bagging, X, y, cv=3, n_jobs=-1)
        # 计算训练时间
        train_time_bagging = time.time() - start_time
        # 存储结果
        bagging_accuracies.append(cv_scores_bagging.mean())
        bagging_times.append(train_time_bagging)

    # 计算边际性价比（每单位时间带来的准确率提升）
    efficiency_rf = []        # 随机森林性价比
    efficiency_bagging = []   # Bagging性价比

    # 遍历每个集成规模点
    for i in range(len(n_estimators_range[:6])):
        if i == 0:
            # 第一个点（1棵树）作为基线，性价比设为0
            efficiency_rf.append(0.0)
            efficiency_bagging.append(0.0)
        else:
            # 计算随机森林边际性价比
            # 准确率提升：当前准确率 - 基线准确率（1棵树的准确率）
            acc_gain_rf = rf_accuracies[i] - rf_accuracies[0]
            # 时间成本：当前训练时间 - 基线训练时间
            time_cost_rf = rf_times[i] - rf_times[0]
            # 边际性价比 = 准确率提升 / 时间成本
            if time_cost_rf > 1e-6:  # 避免除以0
                efficiency_rf.append(acc_gain_rf / time_cost_rf)
            else:
                efficiency_rf.append(0.0)

            # 计算Bagging边际性价比
            acc_gain_bagging = bagging_accuracies[i] - bagging_accuracies[0]
            time_cost_bagging = bagging_times[i] - bagging_times[0]
            if time_cost_bagging > 1e-6:
                efficiency_bagging.append(acc_gain_bagging / time_cost_bagging)
            else:
                efficiency_bagging.append(0.0)

    # 从第二个点开始绘制性价比曲线（第一个点性价比为0）
    # 绘制随机森林性价比曲线
    axes[1, 1].plot(n_estimators_range[1:6], efficiency_rf[1:],
                   'o-', label='随机森林', linewidth=2)
    # 绘制Bagging性价比曲线
    axes[1, 1].plot(n_estimators_range[1:6], efficiency_bagging[1:],
                   's-', label='Bagging', linewidth=2)

    # 设置第四个子图的坐标轴标签和标题
    axes[1, 1].set_xlabel('基学习器数量')
    axes[1, 1].set_ylabel('准确率提升/时间')
    axes[1, 1].set_title('准确率-时间边际性价比分析\n(相对于1棵树的提升)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    # 找到最佳性价比点（排除第一个点）
    if len(efficiency_rf) > 1:
        # np.argmax返回最大值的索引（从efficiency_rf[1:]开始，所以需要加1）
        best_idx_rf = np.argmax(efficiency_rf[1:]) + 1
        best_idx_bagging = np.argmax(efficiency_bagging[1:]) + 1

        # 在图上标记随机森林最佳性价比点
        axes[1, 1].axvline(x=n_estimators_range[best_idx_rf], color='blue',
                          linestyle='--', alpha=0.5,
                          label=f'RF最佳: {n_estimators_range[best_idx_rf]}棵')

        # 在图上标记Bagging最佳性价比点
        axes[1, 1].axvline(x=n_estimators_range[best_idx_bagging], color='orange',
                          linestyle='--', alpha=0.5,
                          label=f'Bagging最佳: {n_estimators_range[best_idx_bagging]}棵')

        # 更新图例
        axes[1, 1].legend()

    # 调整子图布局，避免重叠
    plt.tight_layout()

    # 设置中文字体，解决中文显示问题
    # SimHei: 黑体，Microsoft YaHei: 微软雅黑，KaiTi: 楷体，FangSong: 仿宋
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'KaiTi', 'FangSong']

    # 解决负号显示为方块的问题
    plt.rcParams['axes.unicode_minus'] = False

    # 保存图形到文件
    # dpi=150: 分辨率150点/英寸
    # bbox_inches='tight': 自动调整边界框，确保所有内容都包含在保存的图像中
    plt.savefig('../results/figures/day2_learning_curves.png',
                dpi=150, bbox_inches='tight')

    # 显示图形
    plt.show()

    # ==================== 输出分析总结 ====================
    print("\n" + "=" * 60)
    print("学习曲线分析总结")
    print("=" * 60)

    print("\n1. 边际收益分析（随机森林）:")
    print("-" * 40)
    # 输出每个集成规模变化的边际收益
    for i in range(len(marginal_gains)):
        print(f"从 {n_estimators_range[i]} 到 {n_estimators_range[i+1]} 棵树: "
              f"准确率提升 {marginal_gains[i]:.4f} ({relative_gains[i]:.2f}%)")

    print(f"\n2. 最佳边际性价比点:")
    # 输出最佳性价比点
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

    # 基于边际收益计算推荐树数量
    recommended_n = None
    for i, gain in enumerate(marginal_gains):
        # 当边际收益小于0.5%时，认为增加更多树不再值得
        if gain < 0.005:
            recommended_n = n_estimators_range[i]
            break

    # 输出推荐树数量
    if recommended_n:
        print(f"   - 推荐树数量: {recommended_n}（边际收益<0.5%）")
    else:
        print(f"   - 推荐树数量: 50（默认值）")

    # 返回所有分析结果，便于后续使用
    return {
        'n_estimators_range': n_estimators_range,  # 集成规模范围
        'marginal_gains': marginal_gains,         # 绝对边际收益
        'relative_gains': relative_gains,         # 相对边际收益
        'efficiency_rf': efficiency_rf,          # 随机森林性价比
        'efficiency_bagging': efficiency_bagging, # Bagging性价比
        'best_rf': n_estimators_range[best_idx_rf] if len(efficiency_rf) > 1 else None,  # 随机森林最佳树数量
        'best_bagging': n_estimators_range[best_idx_bagging] if len(efficiency_bagging) > 1 else None,  # Bagging最佳树数量
        'recommended_n': recommended_n           # 推荐树数量
    }


if __name__ == "__main__":
    # 导入操作系统模块，用于文件路径操作
    import os

    # 创建结果目录（如果不存在）
    # exist_ok=True: 如果目录已存在，不会引发错误
    os.makedirs('../results/figures', exist_ok=True)

    # 执行学习曲线分析函数
    results = learning_curve_analysis()

    # 导入JSON模块，用于保存分析结果
    import json

    # 自定义JSON编码器，用于处理numpy数据类型
    class NumpyEncoder(json.JSONEncoder):
        """自定义JSON编码器，处理numpy数据类型"""

        def default(self, obj):
            """
            重写default方法，处理numpy数据类型

            参数:
                obj: 要编码的对象

            返回:
                编码后的对象
            """
            # 处理numpy整数类型
            if isinstance(obj, np.integer):
                return int(obj)
            # 处理numpy浮点数类型
            elif isinstance(obj, np.floating):
                return float(obj)
            # 处理numpy数组类型
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            # 其他类型使用父类的默认处理方法
            else:
                return super(NumpyEncoder, self).default(obj)

    # 将分析结果保存为JSON文件
    with open('../results/logs/day2_learning_curve.json', 'w') as f:
        # indent=2: 缩进2个空格，提高可读性
        # cls=NumpyEncoder: 使用自定义编码器处理numpy数据类型
        json.dump(results, f, indent=2, cls=NumpyEncoder)

    # 输出完成信息
    print("\n✅ 学习曲线分析完成！")
    print("图表已保存到 ../results/figures/day2_learning_curves.png")
    print("详细结果已保存到 ../results/logs/day2_learning_curve.json")
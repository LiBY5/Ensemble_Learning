"""OOB深度分析 - experiments/oob_deep_analysis.py"""

# 导入数值计算库，用于数组操作和数学计算
import numpy as np

# 导入绘图库，用于数据可视化
import matplotlib.pyplot as plt

# 从scikit-learn导入数据生成函数和内置数据集
from sklearn.datasets import make_classification, load_breast_cancer, load_digits

# 从scikit-learn导入随机森林分类器
from sklearn.ensemble import RandomForestClassifier

# 从scikit-learn导入数据分割和交叉验证函数
from sklearn.model_selection import train_test_split, cross_val_score

# 从scikit-learn导入准确率评估指标
from sklearn.metrics import accuracy_score

# 导入警告模块，用于控制警告信息的显示
import warnings

# 设置过滤警告，忽略所有警告信息
warnings.filterwarnings('ignore')


def oob_error_comprehensive_analysis():
    """
    OOB误差全面分析

    该函数执行完整的袋外(OOB)误差分析，包括：
    1. OOB误差与测试误差的对比
    2. OOB误差的统计特性
    3. OOB作为早期停止标准
    4. 不同数据集上的OOB可靠性
    5. OOB误差置信区间分析

    返回:
        dict: 包含所有分析结果的字典
    """

    # 打印标题和分隔线，增强输出可读性
    print("=" * 60)
    print("OOB误差全面分析")
    print("=" * 60)

    # 创建2行3列的子图布局，总图形大小为18x12英寸
    # fig是整体图形对象，axes是包含6个子图对象的2x3数组
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # 为整个图形添加总标题，字体大小为16，加粗显示
    fig.suptitle('OOB误差深度分析', fontsize=16, fontweight='bold')

    # ==================== 分析1: OOB误差与测试误差的对比 ====================
    print("\n分析1: OOB误差与测试误差的对比")

    # 生成模拟分类数据集
    X, y = make_classification(
        n_samples=1000,      # 生成1000个样本
        n_features=20,       # 每个样本有20个特征
        n_informative=15,    # 其中15个是有效特征
        n_classes=2,         # 二分类问题
        random_state=42      # 随机种子，确保结果可复现
    )

    # 将数据集分割为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,                 # 特征和目标变量
        test_size=0.2,       # 20%数据作为测试集
        random_state=42      # 随机种子
    )

    # 定义集成规模（基学习器数量）的范围
    # 从1到200棵树，每隔10棵树测试一次
    n_estimators_range = list(range(1, 201, 10))

    # 初始化存储列表
    oob_errors = []    # 存储OOB误差
    test_errors = []   # 存储测试误差

    # 遍历不同的集成规模
    for n in n_estimators_range:
        # 创建随机森林分类器
        rf = RandomForestClassifier(
            n_estimators=n,     # 树的数量
            max_depth=5,        # 树的最大深度
            oob_score=True,     # 启用OOB评估
            random_state=42,    # 随机种子
            n_jobs=-1          # 使用所有可用的CPU核心
        )
        # 训练模型
        rf.fit(X_train, y_train)

        # 计算OOB误差（1 - OOB分数）
        oob_error = 1 - rf.oob_score_
        oob_errors.append(oob_error)

        # 计算测试误差（1 - 测试准确率）
        test_error = 1 - rf.score(X_test, y_test)
        test_errors.append(test_error)

    # 子图1: OOB误差 vs 测试误差
    ax1 = axes[0, 0]
    # 绘制OOB误差曲线，蓝色实线，线宽2，透明度0.8
    ax1.plot(n_estimators_range, oob_errors, 'b-', label='OOB误差', linewidth=2, alpha=0.8)
    # 绘制测试误差曲线，红色实线，线宽2，透明度0.8
    ax1.plot(n_estimators_range, test_errors, 'r-', label='测试误差', linewidth=2, alpha=0.8)
    # 填充两条曲线之间的区域，灰色，透明度0.2
    ax1.fill_between(n_estimators_range, oob_errors, test_errors, alpha=0.2, color='gray')
    # 设置坐标轴标签
    ax1.set_xlabel('树的数量')
    ax1.set_ylabel('误差')
    # 设置子图标题
    ax1.set_title('OOB误差 vs 测试误差')
    # 显示图例
    ax1.legend()
    # 显示网格，透明度0.3
    ax1.grid(True, alpha=0.3)

    # 计算OOB误差与测试误差之间的相关性
    # np.corrcoef计算相关系数矩阵，[0, 1]获取OOB误差与测试误差之间的相关系数
    correlation = np.corrcoef(oob_errors, test_errors)[0, 1]

    # 在子图上添加文本，显示相关性系数
    ax1.text(0.05, 0.95, f'相关性: {correlation:.3f}',
             transform=ax1.transAxes,  # 使用坐标轴相对坐标
             fontsize=10,              # 字体大小10
             verticalalignment='top',  # 垂直顶部对齐
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))  # 文本框样式

    # 子图2: 误差差异分布
    ax2 = axes[0, 1]
    # 计算OOB误差与测试误差之间的差异
    error_differences = np.array(oob_errors) - np.array(test_errors)
    # 绘制误差差异的直方图，20个柱子，天蓝色，黑色边框，透明度0.7
    ax2.hist(error_differences, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
    # 添加均值线，红色虚线
    ax2.axvline(x=np.mean(error_differences), color='red', linestyle='--',
                label=f'均值: {np.mean(error_differences):.4f}')
    # 添加零线，绿色实线，透明度0.5
    ax2.axvline(x=0, color='green', linestyle='-', alpha=0.5)
    # 设置坐标轴标签
    ax2.set_xlabel('OOB误差 - 测试误差')
    ax2.set_ylabel('频数')
    ax2.set_title('误差差异分布')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # ==================== 分析2: OOB误差的统计特性 ====================
    print("\n分析2: OOB误差的统计特性")

    # 多次实验观察OOB误差的稳定性
    n_repeats = 20  # 重复实验次数
    oob_stability = []   # 存储每次实验的OOB误差
    test_stability = []  # 存储每次实验的测试误差

    # 进行多次重复实验
    for _ in range(n_repeats):
        # 每次实验使用不同的随机种子生成数据
        X, y = make_classification(
            n_samples=500,      # 500个样本
            n_features=10,      # 10个特征
            n_informative=8,    # 8个有效特征
            n_classes=2,        # 二分类
            random_state=np.random.randint(1000)  # 随机种子
        )

        # 分割数据集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # 创建随机森林分类器
        rf = RandomForestClassifier(
            n_estimators=50,   # 50棵树
            oob_score=True,    # 启用OOB评估
            random_state=42    # 随机种子
        )
        # 训练模型
        rf.fit(X_train, y_train)

        # 记录OOB误差和测试误差
        oob_stability.append(1 - rf.oob_score_)
        test_stability.append(1 - rf.score(X_test, y_test))

    # 子图3: OOB误差的稳定性
    ax3 = axes[0, 2]
    x_pos = range(n_repeats)  # x轴位置
    width = 0.35  # 柱状图宽度

    # 绘制OOB误差柱状图，蓝色，透明度0.7
    ax3.bar([x - width / 2 for x in x_pos], oob_stability, width,
            label='OOB误差', color='blue', alpha=0.7)
    # 绘制测试误差柱状图，红色，透明度0.7
    ax3.bar([x + width / 2 for x in x_pos], test_stability, width,
            label='测试误差', color='red', alpha=0.7)
    # 设置坐标轴标签
    ax3.set_xlabel('实验序号')
    ax3.set_ylabel('误差')
    ax3.set_title('OOB误差稳定性分析')
    ax3.legend()
    # 仅在y轴显示网格
    ax3.grid(True, alpha=0.3, axis='y')

    # ==================== 分析3: OOB作为早期停止标准 ====================
    print("\n分析3: OOB作为早期停止标准")

    # 使用更大的树数量范围来观察OOB误差的变化
    n_estimators_large = list(range(1, 501, 20))
    oob_errors_large = []   # 存储OOB误差
    test_errors_large = []  # 存储测试误差

    # 遍历不同的树数量
    for n in n_estimators_large:
        # 创建随机森林分类器
        rf = RandomForestClassifier(
            n_estimators=n,     # 树的数量
            oob_score=True,     # 启用OOB评估
            random_state=42,    # 随机种子
            n_jobs=-1          # 使用所有CPU核心
        )
        # 训练模型
        rf.fit(X_train, y_train)

        # 记录OOB误差和测试误差
        oob_errors_large.append(1 - rf.oob_score_)
        test_errors_large.append(1 - rf.score(X_test, y_test))

    # 子图4: OOB早期停止分析
    ax4 = axes[1, 0]
    # 绘制OOB误差曲线，蓝色实线，线宽2
    ax4.plot(n_estimators_large, oob_errors_large, 'b-', label='OOB误差', linewidth=2)
    # 绘制测试误差曲线，红色实线，线宽2
    ax4.plot(n_estimators_large, test_errors_large, 'r-', label='测试误差', linewidth=2)

    # 找到OOB误差最小的点
    min_oob_idx = np.argmin(oob_errors_large)  # OOB误差最小值的索引
    min_oob_n = n_estimators_large[min_oob_idx]  # 对应的树数量
    min_oob_error = oob_errors_large[min_oob_idx]  # 最小的OOB误差

    # 找到测试误差最小的点
    min_test_idx = np.argmin(test_errors_large)  # 测试误差最小值的索引
    min_test_n = n_estimators_large[min_test_idx]  # 对应的树数量
    min_test_error = test_errors_large[min_test_idx]  # 最小的测试误差

    # 添加OOB最优点的垂直线，蓝色虚线，透明度0.7
    ax4.axvline(x=min_oob_n, color='blue', linestyle='--', alpha=0.7,
                label=f'OOB最优: {min_oob_n}树')
    # 添加测试最优点的垂直线，红色虚线，透明度0.7
    ax4.axvline(x=min_test_n, color='red', linestyle='--', alpha=0.7,
                label=f'测试最优: {min_test_n}树')

    # 设置坐标轴标签
    ax4.set_xlabel('树的数量')
    ax4.set_ylabel('误差')
    ax4.set_title('OOB作为早期停止标准')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # ==================== 分析4: 不同数据集上的OOB可靠性 ====================
    print("\n分析4: 不同数据集上的OOB可靠性")

    # 定义要分析的数据集
    datasets = {
        '模拟数据': make_classification(n_samples=1000, n_features=20,
                                        n_informative=15, random_state=42),
        '乳腺癌': load_breast_cancer(return_X_y=True),  # 加载乳腺癌数据集
        '手写数字': load_digits(return_X_y=True)        # 加载手写数字数据集
    }

    # 初始化存储列表
    dataset_names = []           # 数据集名称列表
    oob_test_correlations = []   # OOB与测试误差的相关性
    oob_test_mae = []            # 平均绝对误差

    # 遍历每个数据集
    for name, (X, y) in datasets.items():
        # 对于手写数字数据集，进行采样以加快计算速度
        if name == '手写数字':
            # 从数据集中随机选择1000个样本（如果数据集大于1000个样本）
            idx = np.random.choice(len(X), 1000, replace=False)
            X, y = X[idx], y[idx]

        # 分割数据集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # 定义要测试的树数量范围
        n_trees_range = [10, 20, 30, 50, 100]
        oob_errors_ds = []  # 存储当前数据集的OOB误差
        test_errors_ds = []  # 存储当前数据集的测试误差

        # 遍历不同的树数量
        for n in n_trees_range:
            # 创建随机森林分类器
            rf = RandomForestClassifier(
                n_estimators=n,     # 树的数量
                oob_score=True,     # 启用OOB评估
                random_state=42,    # 随机种子
                n_jobs=-1          # 使用所有CPU核心
            )
            # 训练模型
            rf.fit(X_train, y_train)

            # 记录OOB误差和测试误差
            oob_errors_ds.append(1 - rf.oob_score_)
            test_errors_ds.append(1 - rf.score(X_test, y_test))

        # 计算OOB误差与测试误差之间的相关性
        if len(oob_errors_ds) > 1:
            # 计算相关系数
            corr = np.corrcoef(oob_errors_ds, test_errors_ds)[0, 1]
            # 计算平均绝对误差
            mae = np.mean(np.abs(np.array(oob_errors_ds) - np.array(test_errors_ds)))
        else:
            corr = 0
            mae = 0

        # 存储结果
        dataset_names.append(name)
        oob_test_correlations.append(corr)
        oob_test_mae.append(mae)

    # 子图5: 不同数据集的OOB可靠性
    ax5 = axes[1, 1]
    x_pos = range(len(dataset_names))  # x轴位置

    # 绘制相关性柱状图，浅绿色，黑色边框，透明度0.7
    bars1 = ax5.bar(x_pos, oob_test_correlations, color='lightgreen',
                    edgecolor='black', alpha=0.7)

    # 设置坐标轴标签
    ax5.set_xlabel('数据集')
    ax5.set_ylabel('相关性')
    ax5.set_title('不同数据集上OOB与测试误差的相关性')
    # 设置x轴刻度
    ax5.set_xticks(x_pos)
    # 设置x轴刻度标签，旋转45度
    ax5.set_xticklabels(dataset_names, rotation=45)
    # 添加高可靠性阈值线（相关性>0.9），红色虚线，透明度0.5
    ax5.axhline(y=0.9, color='red', linestyle='--', alpha=0.5, label='高可靠性阈值')
    ax5.legend()
    # 仅在y轴显示网格
    ax5.grid(True, alpha=0.3, axis='y')

    # 在柱状图上添加数值标签
    for bar, corr in zip(bars1, oob_test_correlations):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                 f'{corr:.3f}', ha='center', va='bottom', fontsize=9)

    # ==================== 分析5: OOB误差置信区间 ====================
    print("\n分析5: OOB误差置信区间分析")

    # 通过Bootstrap方法计算OOB误差的置信区间
    n_bootstraps = 1000  # Bootstrap抽样次数
    oob_bootstrap_errors = []  # 存储Bootstrap样本的OOB误差

    # 生成用于Bootstrap分析的数据集
    X, y = make_classification(n_samples=500, n_features=15,
                               n_informative=10, random_state=42)

    # 进行Bootstrap抽样
    for _ in range(n_bootstraps):
        # 从原始数据集中有放回地随机抽样，生成Bootstrap样本
        indices = np.random.choice(len(X), len(X), replace=True)
        X_boot, y_boot = X[indices], y[indices]

        # 创建随机森林分类器
        rf = RandomForestClassifier(
            n_estimators=50,   # 50棵树
            oob_score=True,    # 启用OOB评估
            random_state=42    # 随机种子
        )
        # 在Bootstrap样本上训练模型
        rf.fit(X_boot, y_boot)

        # 记录OOB误差
        oob_bootstrap_errors.append(1 - rf.oob_score_)

    # 计算OOB误差的统计量
    oob_mean = np.mean(oob_bootstrap_errors)  # 均值
    oob_std = np.std(oob_bootstrap_errors)    # 标准差
    # 计算95%置信区间
    ci_lower = np.percentile(oob_bootstrap_errors, 2.5)   # 下界（2.5%分位数）
    ci_upper = np.percentile(oob_bootstrap_errors, 97.5)  # 上界（97.5%分位数）

    # 子图6: OOB误差置信区间
    ax6 = axes[1, 2]

    # 绘制OOB误差的直方图，30个柱子，天蓝色，黑色边框，透明度0.7，标准化为密度
    ax6.hist(oob_bootstrap_errors, bins=30, color='lightblue',
             edgecolor='black', alpha=0.7, density=True)

    # 导入正态分布函数
    from scipy.stats import norm

    # 生成x轴数据
    x = np.linspace(ci_lower - 0.1, ci_upper + 0.1, 1000)
    # 绘制正态分布曲线，红色实线
    ax6.plot(x, norm.pdf(x, oob_mean, oob_std), 'r-',
             label=f'正态分布\nμ={oob_mean:.3f}, σ={oob_std:.3f}')

    # 添加置信区间下界线，绿色虚线
    ax6.axvline(x=ci_lower, color='green', linestyle='--',
                label=f'95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]')
    # 添加置信区间上界线，绿色虚线
    ax6.axvline(x=ci_upper, color='green', linestyle='--')
    # 添加均值线，蓝色实线
    ax6.axvline(x=oob_mean, color='blue', linestyle='-',
                label=f'均值: {oob_mean:.3f}')

    # 设置坐标轴标签
    ax6.set_xlabel('OOB误差')
    ax6.set_ylabel('密度')
    ax6.set_title('OOB误差Bootstrap置信区间')
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    # 设置中文字体，解决中文显示问题
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'KaiTi', 'FangSong']
    # 解决负号显示为方块的问题
    plt.rcParams['axes.unicode_minus'] = False

    # 调整子图布局，避免重叠
    plt.tight_layout()

    # 保存图形到文件
    plt.savefig('../results/figures/day2_oob_deep_analysis.png', dpi=150, bbox_inches='tight')

    # 显示图形
    plt.show()

    # ==================== 输出详细分析结果 ====================
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
        # 根据相关性评估可靠性
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

    # 返回所有分析结果
    return {
        'correlation': correlation,  # OOB与测试误差的相关性
        'error_differences_mean': np.mean(np.abs(error_differences)),  # 误差差异的平均绝对值
        'oob_stability_mean': np.mean(oob_stability),  # OOB稳定性均值
        'oob_stability_std': np.std(oob_stability),    # OOB稳定性标准差
        'optimal_n_oob': min_oob_n,    # OOB推荐的最优树数量
        'optimal_n_test': min_test_n,  # 测试集推荐的最优树数量
        'dataset_correlations': dict(zip(dataset_names, oob_test_correlations)),  # 不同数据集的相关性
        'bootstrap_ci': [ci_lower, ci_upper, ci_upper - ci_lower]  # Bootstrap置信区间
    }


def oob_statistical_properties():
    """
    分析OOB误差的统计特性

    该函数分析OOB误差的统计特性，包括：
    1. 不同树数量下的OOB误差均值、标准差
    2. OOB误差的偏度和峰度
    3. OOB样本比例

    返回:
        dict: 包含统计特性分析结果的字典
    """
    print("\n" + "=" * 60)
    print("OOB误差统计特性分析")
    print("=" * 60)

    # 生成较大数据集进行统计特性分析
    X, y = make_classification(
        n_samples=2000,       # 2000个样本
        n_features=20,        # 20个特征
        n_informative=15,     # 15个有效特征
        n_classes=2,          # 二分类
        random_state=42       # 随机种子
    )

    # 分析不同树数量下的OOB统计特性
    n_estimators_options = [10, 20, 50, 100, 200]

    # 打印表头
    print(f"{'树数量':<10} {'OOB均值':<10} {'OOB标准差':<10} {'偏度':<10} {'峰度':<10} {'有效样本比例':<15}")
    print("-" * 70)

    # 存储结果
    results = {}

    # 遍历不同的树数量
    for n in n_estimators_options:
        # 创建随机森林分类器
        rf = RandomForestClassifier(
            n_estimators=n,     # 树的数量
            oob_score=True,     # 启用OOB评估
            random_state=42,    # 随机种子
            n_jobs=-1          # 使用所有CPU核心
        )
        # 训练模型
        rf.fit(X, y)

        # 计算理论OOB比例
        # 每个样本是OOB的概率 = (1 - 1/n)^n
        theoretical_oob_ratio = (1 - 1 / n) ** n

        # 检查模型是否有OOB决策函数属性
        if hasattr(rf, 'oob_decision_function_'):
            # 获取OOB决策函数
            oob_decision = rf.oob_decision_function_

            # 计算实际OOB样本比例（非NaN值的比例）
            actual_oob_ratio = np.sum(~np.isnan(oob_decision[:, 0])) / X.shape[0]

            # 计算OOB预测的准确率
            # 从OOB决策函数中获取预测类别
            oob_predictions = np.argmax(oob_decision, axis=1)
            # 筛选出有OOB预测的样本
            valid_indices = ~np.isnan(oob_decision[:, 0])
            # 计算OOB误差
            oob_correct = oob_predictions[valid_indices] == y[valid_indices]
            oob_error = 1 - np.mean(oob_correct)

            # 导入偏度和峰度计算函数
            from scipy.stats import skew, kurtosis
            oob_errors_per_sample = []

            # 计算每个OOB样本的预测误差
            for i in range(X.shape[0]):
                if not np.isnan(oob_decision[i, 0]):
                    true_class = y[i]
                    # 获取真实类别的预测概率
                    confidence = oob_decision[i, true_class]
                    # 计算误差（1 - 置信度）
                    oob_errors_per_sample.append(1 - confidence)

            # 计算统计特性
            if len(oob_errors_per_sample) > 0:
                skewness = skew(oob_errors_per_sample)  # 偏度
                kurt = kurtosis(oob_errors_per_sample)  # 峰度
                std_dev = np.std(oob_errors_per_sample)  # 标准差
            else:
                skewness = kurt = std_dev = np.nan

            # 打印结果
            print(
                f"{n:<10} {oob_error:<10.4f} {std_dev:<10.4f} "
                f"{skewness:<10.4f} {kurt:<10.4f} {actual_oob_ratio:<15.4f}")

            # 存储结果
            results[n] = {
                'oob_error': oob_error,  # OOB误差
                'std': std_dev,          # 标准差
                'skewness': skewness,    # 偏度
                'kurtosis': kurt,        # 峰度
                'actual_oob_ratio': actual_oob_ratio,        # 实际OOB比例
                'theoretical_oob_ratio': theoretical_oob_ratio  # 理论OOB比例
            }
        else:
            # 如果模型没有OOB决策函数属性，打印N/A
            print(f"{n:<10} {'N/A':<10} {'N/A':<10} {'N/A':<10} {'N/A':<10} {'N/A':<15}")

    return results


if __name__ == "__main__":
    # 导入操作系统模块，用于文件路径操作
    import os

    # 创建结果目录（如果不存在）
    os.makedirs('../results/figures', exist_ok=True)

    # 运行OOB全面分析
    analysis_results = oob_error_comprehensive_analysis()

    # 运行OOB统计特性分析
    stat_results = oob_statistical_properties()

    # 保存结果到JSON文件
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

    # 合并所有结果
    all_results = {
        'comprehensive_analysis': analysis_results,
        'statistical_properties': stat_results
    }

    # 将分析结果保存为JSON文件
    with open('../results/logs/day2_oob_analysis.json', 'w') as f:
        json.dump(all_results, f, indent=2, cls=NumpyEncoder)

    # 输出完成信息
    print("\n✅ OOB深度分析完成！")
    print("图表已保存到 ../results/figures/day2_oob_deep_analysis.png")
    print("详细结果已保存到 ../results/logs/day2_oob_analysis.json")
"""投票集成在不同数据集上的表现"""

import numpy as np  # 导入numpy库，用于数值计算
import matplotlib.pyplot as plt  # 导入matplotlib库，用于数据可视化
from sklearn.datasets import load_iris, load_breast_cancer, load_wine, load_digits  # 导入scikit-learn中的数据集
from sklearn.model_selection import train_test_split  # 导入数据集划分函数
import json  # 导入json模块，用于保存和加载JSON格式数据
import os  # 导入os模块，用于操作系统相关功能（如创建目录）

# 从自定义模块导入模型
from models.base import get_diverse_classifiers  # 导入获取多种基分类器的函数
from models.ensemble import VotingEnsemble, WeightedVotingEnsemble  # 导入投票集成和加权投票集成类

# 方法：设置字体为系统自带的中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'KaiTi', 'FangSong']  # 设置中文字体
# 解决负号显示问题
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方块的问题


def run_experiment_on_dataset(dataset_name, voting_method='hard',
                              test_size=0.2, random_state=42):
    """在单个数据集上运行实验"""

    # 加载数据集
    if dataset_name == 'iris':
        data = load_iris()  # 加载鸢尾花数据集
        dataset_desc = "鸢尾花数据集 (3类, 150样本)"  # 数据集描述
    elif dataset_name == 'breast_cancer':
        data = load_breast_cancer()  # 加载乳腺癌数据集
        dataset_desc = "乳腺癌数据集 (2类, 569样本)"  # 数据集描述
    elif dataset_name == 'wine':
        data = load_wine()  # 加载葡萄酒数据集
        dataset_desc = "葡萄酒数据集 (3类, 178样本)"  # 数据集描述
    elif dataset_name == 'digits':
        data = load_digits()  # 加载手写数字数据集
        dataset_desc = "手写数字数据集 (10类, 1797样本)"  # 数据集描述
    else:
        raise ValueError(f"未知数据集: {dataset_name}")  # 如果数据集名称未知则抛出异常

    X, y = data.data, data.target  # 获取特征矩阵和标签向量

    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y  # 按比例划分训练集和测试集，保持类别分布
    )

    # 打印数据集信息
    print(f"\n{'=' * 60}")  # 打印分隔线
    print(f"数据集: {dataset_desc}")  # 打印数据集描述
    print(f"特征数: {X.shape[1]}, 类别数: {len(np.unique(y))}")  # 打印特征数和类别数
    print(f"训练集: {X_train.shape}, 测试集: {X_test.shape}")  # 打印训练集和测试集大小
    print(f"投票方法: {voting_method}")  # 打印投票方法
    print('=' * 60)  # 打印分隔线

    # 获取基分类器
    base_classifiers = get_diverse_classifiers(random_state=random_state)  # 获取多种基分类器

    # 1. 评估单个分类器
    print("\n1. 基分类器性能:")  # 打印标题
    print("-" * 50)  # 打印分隔线

    single_results = {}  # 初始化字典存储单个分类器结果
    for clf in base_classifiers:  # 遍历每个基分类器
        clf.fit(X_train, y_train)  # 训练分类器
        acc = clf.score(X_test, y_test)  # 在测试集上评估准确率
        single_results[clf.name] = float(acc)  # 转换为Python float类型并存储结果
        print(f"  {clf.name:25s}: {acc:.4f}")  # 打印每个分类器的准确率

    # 2. 普通投票集成
    print(f"\n2. {voting_method}投票集成:")  # 打印标题
    print("-" * 50)  # 打印分隔线

    ensemble = VotingEnsemble(base_classifiers, voting=voting_method)  # 创建普通投票集成
    ensemble.fit(X_train, y_train)  # 训练集成模型
    ensemble_acc = float(ensemble.score(X_test, y_test))  # 在测试集上评估准确率
    print(f"  {ensemble.name:25s}: {ensemble_acc:.4f}")  # 打印集成模型准确率

    # 3. 权重优化投票集成
    print(f"\n3. 权重优化投票集成:")  # 打印标题
    print("-" * 50)  # 打印分隔线

    # 划分验证集
    X_train_sub, X_val, y_train_sub, y_val = train_test_split(
        X_train, y_train, test_size=0.25, random_state=random_state, stratify=y_train  # 从训练集中划分验证集
    )

    # 重新训练基分类器
    base_classifiers2 = get_diverse_classifiers(random_state=random_state + 1)  # 获取新的基分类器实例
    for clf in base_classifiers2:  # 遍历每个基分类器
        clf.fit(X_train_sub, y_train_sub)  # 在训练子集上训练

    weighted_ensemble = WeightedVotingEnsemble(base_classifiers2, voting='soft')  # 创建加权投票集成
    weighted_ensemble.optimize_weights(X_val, y_val, metric='accuracy', method='inverse_error')  # 优化权重
    weighted_ensemble.fit(X_train_sub, y_train_sub, verbose=False)  # 训练加权集成模型
    weighted_acc = float(weighted_ensemble.score(X_test, y_test))  # 在测试集上评估准确率
    print(f"  {weighted_ensemble.name:25s}: {weighted_acc:.4f}")  # 打印加权集成准确率

    # 4. 性能对比
    best_single_name = max(single_results, key=single_results.get)  # 找到最佳单模型名称
    best_single_acc = single_results[best_single_name]  # 获取最佳单模型准确率

    # 计算提升比例
    improvement_plain = (ensemble_acc - best_single_acc) / best_single_acc * 100  # 普通集成相对提升
    improvement_weighted = (weighted_acc - best_single_acc) / best_single_acc * 100  # 加权集成相对提升

    print(f"\n4. 结果总结:")  # 打印标题
    print("-" * 50)  # 打印分隔线
    print(f"  最佳单模型: {best_single_name} ({best_single_acc:.4f})")  # 打印最佳单模型信息
    print(f"  普通投票集成: {ensemble_acc:.4f} ({improvement_plain:+.2f}%)")  # 打印普通集成结果
    print(f"  加权投票集成: {weighted_acc:.4f} ({improvement_weighted:+.2f}%)")  # 打印加权集成结果

    # 返回实验结果
    return {
        'dataset': dataset_name,  # 数据集名称
        'dataset_description': dataset_desc,  # 数据集描述
        'n_samples': X.shape[0],  # 样本数量
        'n_features': X.shape[1],  # 特征数量
        'n_classes': len(np.unique(y)),  # 类别数量
        'single_results': single_results,  # 单个分类器结果
        'ensemble_accuracy': ensemble_acc,  # 普通集成准确率
        'weighted_ensemble_accuracy': weighted_acc,  # 加权集成准确率
        'best_single': {'name': best_single_name, 'accuracy': best_single_acc},  # 最佳单模型信息
        'improvement_plain': improvement_plain,  # 普通集成提升比例
        'improvement_weighted': improvement_weighted  # 加权集成提升比例
    }


def visualize_results(all_results, save_dir='C:/Users/l/Desktop/L/python/ensemble_learning/results/figures'):
    """可视化所有实验结果"""

    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)  # 创建保存目录，如果目录不存在

    datasets = list(all_results.keys())  # 获取所有数据集名称
    n_datasets = len(datasets)  # 数据集数量

    # 创建子图
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))  # 创建2x2的子图
    axes = axes.flat  # 将子图数组展平为一维

    # 对每个数据集绘制结果
    for idx, dataset_name in enumerate(datasets):
        if idx >= 4:  # 最多显示4个子图
            break

        ax = axes[idx]  # 获取当前子图
        result = all_results[dataset_name]  # 获取当前数据集的结果

        # 准备数据
        single_results = result['single_results']  # 获取单个分类器结果
        model_names = list(single_results.keys())  # 获取模型名称列表
        single_accs = list(single_results.values())  # 获取模型准确率列表

        # 添加集成结果
        model_names.append('VotingEnsemble')  # 添加普通投票集成名称
        model_names.append('WeightedVoting')  # 添加加权投票集成名称

        single_accs.append(result['ensemble_accuracy'])  # 添加普通集成准确率
        single_accs.append(result['weighted_ensemble_accuracy'])  # 添加加权集成准确率

        # 颜色设置
        colors = ['lightblue'] * len(result['single_results']) + ['lightgreen', 'orange']  # 设置不同模型的颜色

        # 创建条形图
        bars = ax.bar(range(len(model_names)), single_accs, color=colors, edgecolor='black')  # 绘制条形图

        # 添加数值标签
        for bar, acc in zip(bars, single_accs):
            height = bar.get_height()  # 获取条形高度
            ax.text(bar.get_x() + bar.get_width() / 2., height + 0.005,
                    f'{acc:.3f}', ha='center', va='bottom', fontsize=8)  # 在条形上方添加准确率文本

        # 设置图表属性
        ax.set_xticks(range(len(model_names)))  # 设置x轴刻度
        ax.set_xticklabels(model_names, rotation=45, ha='right', fontsize=9)  # 设置x轴标签
        ax.set_ylabel('准确率', fontsize=10)  # 设置y轴标签
        ax.set_title(f'{dataset_name}数据集', fontsize=12, fontweight='bold')  # 设置图表标题
        ax.set_ylim([min(single_accs) * 0.95, 1.0])  # 设置y轴范围
        ax.grid(True, alpha=0.3, axis='y')  # 添加网格线

        # 添加参考线
        best_single = result['best_single']['accuracy']  # 获取最佳单模型准确率
        ax.axhline(y=best_single, color='red', linestyle='--', alpha=0.7,
                   label=f"最佳单模型: {best_single:.3f}")  # 添加水平参考线
        ax.legend(fontsize=9)  # 添加图例

    # 调整布局
    plt.suptitle('投票集成在不同数据集上的性能对比', fontsize=16, fontweight='bold', y=1.02)  # 添加总标题
    plt.tight_layout()  # 调整子图布局

    # 保存图表
    plt.savefig(f'{save_dir}/day1_voting_results.png', dpi=150, bbox_inches='tight')  # 保存图表为PNG文件
    plt.show()  # 显示图表

    # 创建汇总图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))  # 创建包含两个子图的图形

    # 汇总准确率
    dataset_labels = []  # 初始化数据集标签列表
    best_single_accs = []  # 初始化最佳单模型准确率列表
    ensemble_accs = []  # 初始化普通集成准确率列表
    weighted_accs = []  # 初始化加权集成准确率列表

    # 收集各数据集的结果
    for dataset_name, result in all_results.items():
        dataset_labels.append(dataset_name)  # 添加数据集名称
        best_single_accs.append(result['best_single']['accuracy'])  # 添加最佳单模型准确率
        ensemble_accs.append(result['ensemble_accuracy'])  # 添加普通集成准确率
        weighted_accs.append(result['weighted_ensemble_accuracy'])  # 添加加权集成准确率

    x = np.arange(len(dataset_labels))  # 创建x轴位置数组
    width = 0.25  # 设置条形宽度

    # 绘制汇总条形图
    ax1.bar(x - width, best_single_accs, width, label='最佳单模型', color='lightblue', edgecolor='black')
    ax1.bar(x, ensemble_accs, width, label='普通投票集成', color='lightgreen', edgecolor='black')
    ax1.bar(x + width, weighted_accs, width, label='加权投票集成', color='orange', edgecolor='black')

    # 设置第一个子图属性
    ax1.set_xlabel('数据集', fontsize=12)  # 设置x轴标签
    ax1.set_ylabel('准确率', fontsize=12)  # 设置y轴标签
    ax1.set_title('不同方法在各数据集上的准确率', fontsize=14, fontweight='bold')  # 设置标题
    ax1.set_xticks(x)  # 设置x轴刻度
    ax1.set_xticklabels(dataset_labels)  # 设置x轴标签
    ax1.legend()  # 添加图例
    ax1.grid(True, alpha=0.3, axis='y')  # 添加网格线

    # 添加数值标签
    for i, (best, ens, weighted) in enumerate(zip(best_single_accs, ensemble_accs, weighted_accs)):
        ax1.text(i - width, best + 0.01, f'{best:.3f}', ha='center', va='bottom', fontsize=9)
        ax1.text(i, ens + 0.01, f'{ens:.3f}', ha='center', va='bottom', fontsize=9)
        ax1.text(i + width, weighted + 0.01, f'{weighted:.3f}', ha='center', va='bottom', fontsize=9)

    # 汇总提升比例
    improvements_plain = [result['improvement_plain'] for result in all_results.values()]  # 获取普通集成提升比例
    improvements_weighted = [result['improvement_weighted'] for result in all_results.values()]  # 获取加权集成提升比例

    # 绘制提升比例条形图
    ax2.bar(x - width / 2, improvements_plain, width, label='普通集成提升', color='lightgreen', edgecolor='black')
    ax2.bar(x + width / 2, improvements_weighted, width, label='加权集成提升', color='orange', edgecolor='black')

    # 设置第二个子图属性
    ax2.set_xlabel('数据集', fontsize=12)  # 设置x轴标签
    ax2.set_ylabel('相对提升 (%)', fontsize=12)  # 设置y轴标签
    ax2.set_title('集成相对最佳单模型的提升比例', fontsize=14, fontweight='bold')  # 设置标题
    ax2.set_xticks(x)  # 设置x轴刻度
    ax2.set_xticklabels(dataset_labels)  # 设置x轴标签
    ax2.axhline(y=0, color='black', linewidth=0.8)  # 添加零线
    ax2.legend()  # 添加图例
    ax2.grid(True, alpha=0.3, axis='y')  # 添加网格线

    # 添加数值标签
    for i, (imp_plain, imp_weighted) in enumerate(zip(improvements_plain, improvements_weighted)):
        ax2.text(i - width / 2, imp_plain + (0.5 if imp_plain >= 0 else -0.8),
                 f'{imp_plain:+.1f}%', ha='center', va='bottom' if imp_plain >= 0 else 'top', fontsize=9)
        ax2.text(i + width / 2, imp_weighted + (0.5 if imp_weighted >= 0 else -0.8),
                 f'{imp_weighted:+.1f}%', ha='center', va='bottom' if imp_weighted >= 0 else 'top', fontsize=9)

    plt.tight_layout()  # 调整布局
    plt.savefig(f'{save_dir}/day1_summary_results.png', dpi=150, bbox_inches='tight')  # 保存图表
    plt.show()  # 显示图表


def main():
    """主函数：运行所有实验"""

    # 确保目录存在
    os.makedirs('C:/Users/l/Desktop/L/python/ensemble_learning/results/figures', exist_ok=True)  # 创建图表保存目录
    os.makedirs('C:/Users/l/Desktop/L/python/ensemble_learning/results/logs', exist_ok=True)  # 创建日志保存目录

    # 运行实验的数据集
    datasets = ['iris', 'breast_cancer', 'wine', 'digits']  # 数据集列表
    voting_methods = ['hard', 'soft']  # 投票方法列表

    all_results = {}  # 初始化结果字典

    # 对每个数据集和每种投票方法运行实验
    for dataset in datasets:
        all_results[dataset] = {}  # 为每个数据集初始化结果字典

        for method in voting_methods:  # 遍历每种投票方法
            print(f"\n{'#' * 80}")  # 打印分隔线
            print(f"开始实验: 数据集={dataset}, 投票方法={method}")  # 打印实验信息
            print(f"{'#' * 80}")

            # 运行实验
            result = run_experiment_on_dataset(
                dataset_name=dataset,
                voting_method=method,
                random_state=42
            )

            all_results[dataset][method] = result  # 存储结果

            print(f"\n{'#' * 80}")  # 打印分隔线
            print(f"实验完成: 数据集={dataset}, 投票方法={method}")  # 打印完成信息
            print(f"{'#' * 80}\n")

    # 可视化结果
    print("\n生成可视化结果...")  # 打印提示信息

    # 为每个投票方法单独可视化
    for method in voting_methods:
        method_results = {}  # 初始化方法结果字典
        for dataset in datasets:
            method_results[dataset] = all_results[dataset][method]  # 收集该方法的结果

        # 保存详细结果
        with open(f'C:/Users/l/Desktop/L/python/ensemble_learning/results/logs/day1_{method}_voting_results.json', 'w') as f:
            json.dump(method_results, f, indent=2, ensure_ascii=False)  # 保存为JSON文件

    # 使用软投票的结果进行可视化
    soft_results = {}  # 初始化软投票结果字典
    for dataset in datasets:
        soft_results[dataset] = all_results[dataset]['soft']  # 收集软投票结果

    visualize_results(soft_results)  # 可视化软投票结果

    # 打印汇总结果
    print("\n" + "=" * 80)  # 打印分隔线
    print("实验汇总")  # 打印标题
    print("=" * 80)  # 打印分隔线

    # 打印每个数据集的汇总结果
    for dataset in datasets:
        print(f"\n数据集: {dataset}")  # 打印数据集名称
        print("-" * 40)  # 打印分隔线

        for method in voting_methods:  # 遍历每种投票方法
            data = all_results[dataset][method]  # 获取该数据集该方法的结果
            print(f"  方法: {method}")  # 打印方法名称
            print(f"    最佳单模型: {data['best_single']['name']} ({data['best_single']['accuracy']:.4f})")  # 打印最佳单模型
            print(f"    普通集成: {data['ensemble_accuracy']:.4f} (提升: {data['improvement_plain']:+.2f}%)")  # 打印普通集成结果
            print(f"    加权集成: {data['weighted_ensemble_accuracy']:.4f} (提升: {data['improvement_weighted']:+.2f}%)")  # 打印加权集成结果

    # 保存所有结果
    with open('C:/Users/l/Desktop/L/python/ensemble_learning/results/logs/day1_all_experiment_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)  # 保存所有结果为JSON文件

    # 打印保存信息
    print(f"\n所有实验完成！结果已保存到:")
    print(f"  - C:/Users/l/Desktop/L/python/ensemble_learning/results/logs/day1_all_experiment_results.json")
    print(f"  - C:/Users/l/Desktop/L/python/ensemble_learning/results/figures/day1_voting_results.png")
    print(f"  - C:/Users/l/Desktop/L/python/ensemble_learning/results/figures/day1_summary_results.png")


if __name__ == "__main__":
    main()  # 运行主函数
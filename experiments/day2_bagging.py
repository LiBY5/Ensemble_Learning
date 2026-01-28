"""Bagging和随机森林综合实验 - experiments/day2_bagging_experiments.py"""
import numpy as np  # 导入NumPy库，用于数值计算
import matplotlib.pyplot as plt  # 导入Matplotlib库，用于数据可视化
from sklearn.datasets import load_breast_cancer, load_wine, load_digits  # 导入scikit-learn内置数据集
from sklearn.model_selection import train_test_split, cross_val_score  # 导入数据集划分和交叉验证函数
from sklearn.tree import DecisionTreeClassifier  # 导入决策树分类器
from sklearn.ensemble import BaggingClassifier as SklearnBagging  # 导入scikit-learn的Bagging分类器
from sklearn.ensemble import RandomForestClassifier as SklearnRF  # 导入scikit-learn的随机森林分类器
from sklearn.ensemble import ExtraTreesClassifier  # 导入极端随机树分类器

# 导入我们的实现
import sys  # 导入sys模块，用于操作系统相关功能
import os  # 导入os模块，用于操作系统相关功能

# 将父目录添加到系统路径，以便导入自定义模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 从自定义模块导入Bagging和随机森林分类器
from models.bagging import BaggingClassifier
from models.random_forest import RandomForestClassifier


def compare_ensemble_methods(dataset_name='breast_cancer'):
    """对比不同集成方法"""
    print("\n" + "=" * 60)  # 打印分隔线
    print(f"数据集: {dataset_name}")  # 打印数据集名称
    print("=" * 60)  # 打印分隔线

    # 加载数据集
    if dataset_name == 'breast_cancer':  # 如果数据集名称为乳腺癌数据集
        data = load_breast_cancer()  # 加载乳腺癌数据集
        dataset_info = "乳腺癌数据集（二分类）"  # 设置数据集信息
    elif dataset_name == 'wine':  # 如果数据集名称为红酒数据集
        data = load_wine()  # 加载红酒数据集
        dataset_info = "红酒数据集（多分类）"  # 设置数据集信息
    elif dataset_name == 'digits':  # 如果数据集名称为手写数字数据集
        data = load_digits()  # 加载手写数字数据集
        dataset_info = "手写数字数据集（多分类）"  # 设置数据集信息
    else:  # 如果数据集名称未知
        raise ValueError(f"未知数据集: {dataset_name}")  # 抛出值错误异常

    X, y = data.data, data.target  # 获取特征数据和目标标签

    print(f"数据集信息: {dataset_info}")  # 打印数据集信息
    print(f"样本数: {X.shape[0]}, 特征数: {X.shape[1]}, 类别数: {len(np.unique(y))}")  # 打印数据集统计信息

    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y  # 将数据集划分为训练集和测试集，保持类别分布
    )

    # 基学习器
    base_tree = DecisionTreeClassifier(max_depth=5, random_state=42)  # 创建决策树基学习器

    # 定义所有模型
    models = {  # 创建模型字典，包含各种集成方法
        '决策树': DecisionTreeClassifier(max_depth=5, random_state=42),  # 单棵决策树
        'Bagging (我们的实现)': BaggingClassifier(  # 自定义Bagging分类器
            base_estimator=DecisionTreeClassifier(max_depth=5, random_state=42),  # 基学习器为决策树
            n_estimators=50,  # 基学习器数量为50
            max_samples=0.8,  # 每个基学习器使用的样本比例为80%
            max_features=0.8,  # 每个基学习器使用的特征比例为80%
            oob_score=True,  # 启用袋外样本评估
            random_state=42,  # 随机种子
            verbose=0  # 不输出训练过程信息
        ),
        'Bagging (sklearn)': SklearnBagging(  # scikit-learn的Bagging分类器
            estimator=DecisionTreeClassifier(max_depth=5, random_state=42),  # 基学习器为决策树
            n_estimators=50,  # 基学习器数量为50
            max_samples=0.8,  # 每个基学习器使用的样本比例为80%
            max_features=0.8,  # 每个基学习器使用的特征比例为80%
            oob_score=True,  # 启用袋外样本评估
            random_state=42  # 随机种子
        ),
        '随机森林 (我们的实现)': RandomForestClassifier(  # 自定义随机森林分类器
            n_estimators=50,  # 决策树数量为50
            max_depth=5,  # 决策树最大深度为5
            max_features='sqrt',  # 每个节点分裂时考虑的特征数为sqrt(总特征数)
            oob_score=True,  # 启用袋外样本评估
            random_state=42,  # 随机种子
            verbose=0  # 不输出训练过程信息
        ),
        '随机森林 (sklearn)': SklearnRF(  # scikit-learn的随机森林分类器
            n_estimators=50,  # 决策树数量为50
            max_depth=5,  # 决策树最大深度为5
            max_features='sqrt',  # 每个节点分裂时考虑的特征数为sqrt(总特征数)
            oob_score=True,  # 启用袋外样本评估
            random_state=42,  # 随机种子
            n_jobs=-1  # 使用所有可用的CPU核心进行并行计算
        ),
        '极端随机树 (sklearn)': ExtraTreesClassifier(  # scikit-learn的极端随机树分类器
            n_estimators=50,  # 决策树数量为50
            max_depth=5,  # 决策树最大深度为5
            max_features='sqrt',  # 每个节点分裂时考虑的特征数为sqrt(总特征数)
            random_state=42  # 随机种子
        )
    }

    # 训练和评估
    results = {}  # 创建结果字典

    for name, model in models.items():  # 遍历所有模型
        print(f"\n训练 {name}...")  # 打印当前训练的模型名称

        # 训练
        import time  # 导入时间模块
        start_time = time.time()  # 记录训练开始时间
        model.fit(X_train, y_train)  # 训练模型
        train_time = time.time() - start_time  # 计算训练时间

        # 评估
        train_acc = model.score(X_train, y_train)  # 计算训练集准确率
        test_acc = model.score(X_test, y_test)  # 计算测试集准确率

        # 获取OOB分数（如果可用）
        oob_score = getattr(model, 'oob_score_', None)  # 获取袋外样本评估分数，如果不存在则返回None

        # 交叉验证
        cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy', n_jobs=-1)  # 执行5折交叉验证

        results[name] = {  # 将结果保存到字典中
            'train_accuracy': train_acc,  # 训练集准确率
            'test_accuracy': test_acc,  # 测试集准确率
            'cv_mean': cv_scores.mean(),  # 交叉验证平均准确率
            'cv_std': cv_scores.std(),  # 交叉验证准确率标准差
            'oob_score': oob_score,  # 袋外样本评估分数
            'train_time': train_time  # 训练时间
        }

        print(f"  训练时间: {train_time:.3f}s")  # 打印训练时间
        print(f"  训练准确率: {train_acc:.4f}")  # 打印训练集准确率
        print(f"  测试准确率: {test_acc:.4f}")  # 打印测试集准确率
        print(f"  交叉验证: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")  # 打印交叉验证结果
        if oob_score is not None:  # 如果存在袋外样本评估分数
            print(f"  OOB分数: {oob_score:.4f}")  # 打印袋外样本评估分数

    return results, models, dataset_info  # 返回结果、模型和数据集信息


def visualize_comparison(results, dataset_name, dataset_info):
    """可视化对比结果"""
    names = list(results.keys())  # 获取所有模型名称
    test_accs = [results[name]['test_accuracy'] for name in names]  # 获取所有模型的测试准确率
    cv_means = [results[name]['cv_mean'] for name in names]  # 获取所有模型的交叉验证平均准确率
    cv_stds = [results[name]['cv_std'] for name in names]  # 获取所有模型的交叉验证准确率标准差
    train_times = [results[name]['train_time'] for name in names]  # 获取所有模型的训练时间

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))  # 创建2x2的子图
    fig.suptitle(f'{dataset_info} - 不同集成方法对比', fontsize=16)  # 设置总标题

    x = np.arange(len(names))  # 创建x轴位置数组
    width = 0.35  # 设置柱状图宽度

    # 1. 测试准确率
    bars1 = axes[0, 0].bar(x, test_accs, width, color='skyblue', edgecolor='black')  # 绘制测试准确率柱状图
    axes[0, 0].set_xlabel('模型')  # 设置x轴标签
    axes[0, 0].set_ylabel('测试准确率')  # 设置y轴标签
    axes[0, 0].set_title('测试准确率对比')  # 设置子图标题
    axes[0, 0].set_xticks(x)  # 设置x轴刻度位置
    axes[0, 0].set_xticklabels(names, rotation=45, ha='right')  # 设置x轴刻度标签，旋转45度
    axes[0, 0].grid(True, alpha=0.3, axis='y')  # 显示y轴网格线

    # 添加数值标签
    for bar in bars1:  # 遍历每个柱状图
        height = bar.get_height()  # 获取柱状图高度
        axes[0, 0].annotate(f'{height:.3f}',  # 添加数值标签
                            xy=(bar.get_x() + bar.get_width() / 2, height),  # 标签位置
                            xytext=(0, 3),  # 标签偏移量
                            textcoords="offset points",  # 使用偏移坐标
                            ha='center', va='bottom', fontsize=9)  # 设置标签对齐方式和字体大小

    # 2. 交叉验证结果
    bars2 = axes[0, 1].bar(x, cv_means, yerr=cv_stds, capsize=5,  # 绘制交叉验证结果柱状图，包含误差条
                           color='lightcoral', edgecolor='black', alpha=0.7)  # 设置柱状图颜色和透明度
    axes[0, 1].set_xlabel('模型')  # 设置x轴标签
    axes[0, 1].set_ylabel('准确率')  # 设置y轴标签
    axes[0, 1].set_title('5折交叉验证结果（均值±标准差）')  # 设置子图标题
    axes[0, 1].set_xticks(x)  # 设置x轴刻度位置
    axes[0, 1].set_xticklabels(names, rotation=45, ha='right')  # 设置x轴刻度标签，旋转45度
    axes[0, 1].grid(True, alpha=0.3, axis='y')  # 显示y轴网格线

    # 添加数值标签
    for bar, std in zip(bars2, cv_stds):  # 遍历每个柱状图和对应的标准差
        height = bar.get_height()  # 获取柱状图高度
        axes[0, 1].annotate(f'{height:.3f}±{std:.3f}',  # 添加数值标签，包含标准差
                            xy=(bar.get_x() + bar.get_width() / 2, height),  # 标签位置
                            xytext=(0, 3),  # 标签偏移量
                            textcoords="offset points",  # 使用偏移坐标
                            ha='center', va='bottom', fontsize=8)  # 设置标签对齐方式和字体大小

    # 3. OOB分数（如果有）
    oob_scores = []  # 创建OOB分数列表
    for name in names:  # 遍历所有模型名称
        oob = results[name]['oob_score']  # 获取OOB分数
        oob_scores.append(oob if oob is not None else 0)  # 如果OOB分数存在则添加，否则添加0

    bars3 = axes[1, 0].bar(x, oob_scores, width, color='lightgreen', edgecolor='black')  # 绘制OOB分数柱状图
    axes[1, 0].set_xlabel('模型')  # 设置x轴标签
    axes[1, 0].set_ylabel('OOB分数')  # 设置y轴标签
    axes[1, 0].set_title('OOB分数对比（支持OOB的模型）')  # 设置子图标题
    axes[1, 0].set_xticks(x)  # 设置x轴刻度位置
    axes[1, 0].set_xticklabels(names, rotation=45, ha='right')  # 设置x轴刻度标签，旋转45度
    axes[1, 0].grid(True, alpha=0.3, axis='y')  # 显示y轴网格线

    # 添加数值标签
    for bar, score in zip(bars3, oob_scores):  # 遍历每个柱状图和对应的OOB分数
        if score > 0:  # 如果OOB分数大于0
            height = bar.get_height()  # 获取柱状图高度
            axes[1, 0].annotate(f'{height:.3f}',  # 添加数值标签
                                xy=(bar.get_x() + bar.get_width() / 2, height),  # 标签位置
                                xytext=(0, 3),  # 标签偏移量
                                textcoords="offset points",  # 使用偏移坐标
                                ha='center', va='bottom', fontsize=9)  # 设置标签对齐方式和字体大小

    # 4. 训练时间
    bars4 = axes[1, 1].bar(x, train_times, width, color='gold', edgecolor='black')  # 绘制训练时间柱状图
    axes[1, 1].set_xlabel('模型')  # 设置x轴标签
    axes[1, 1].set_ylabel('训练时间（秒）')  # 设置y轴标签
    axes[1, 1].set_title('训练时间对比')  # 设置子图标题
    axes[1, 1].set_xticks(x)  # 设置x轴刻度位置
    axes[1, 1].set_xticklabels(names, rotation=45, ha='right')  # 设置x轴刻度标签，旋转45度
    axes[1, 1].grid(True, alpha=0.3, axis='y')  # 显示y轴网格线

    # 添加数值标签
    for bar in bars4:  # 遍历每个柱状图
        height = bar.get_height()  # 获取柱状图高度
        axes[1, 1].annotate(f'{height:.3f}s',  # 添加数值标签
                            xy=(bar.get_x() + bar.get_width() / 2, height),  # 标签位置
                            xytext=(0, 3),  # 标签偏移量
                            textcoords="offset points",  # 使用偏移坐标
                            ha='center', va='bottom', fontsize=9)  # 设置标签对齐方式和字体大小

    # 设置字体为系统自带的中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'KaiTi', 'FangSong']  # 设置中文字体
    # 解决负号显示问题
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方块的问题
    plt.tight_layout()  # 自动调整子图参数
    plt.savefig(f'../results/figures/day2_comparison_{dataset_name}.png', dpi=150, bbox_inches='tight')  # 保存图表
    plt.show()  # 显示图表

    # 计算相对于决策树的提升
    baseline = results['决策树']['test_accuracy']  # 获取决策树的测试准确率作为基线
    print(f"\n相对于决策树的提升:")  # 打印提升信息标题
    print("-" * 50)  # 打印分隔线
    for name in names:  # 遍历所有模型名称
        if name != '决策树':  # 如果不是决策树模型
            improvement = (results[name]['test_accuracy'] - baseline) / baseline * 100  # 计算相对于基线的提升百分比
            print(f"{name:25s}: {improvement:6.2f}% (测试准确率: {results[name]['test_accuracy']:.4f})")  # 打印提升信息


def run_all_datasets():
    """在所有数据集上运行实验"""
    datasets = ['breast_cancer', 'wine', 'digits']  # 定义所有数据集名称
    all_results = {}  # 创建所有结果的字典

    for dataset in datasets:  # 遍历所有数据集
        results, models, dataset_info = compare_ensemble_methods(dataset)  # 运行集成方法对比实验
        all_results[dataset] = results  # 将结果保存到字典中

        visualize_comparison(results, dataset, dataset_info)  # 可视化对比结果

    return all_results  # 返回所有结果


if __name__ == "__main__":
    import os  # 导入os模块

    os.makedirs('../results/figures', exist_ok=True)  # 创建结果目录，如果不存在则创建

    print("开始综合对比实验...")  # 打印开始实验信息
    all_results = run_all_datasets()  # 运行所有数据集上的实验

    # 保存结果
    import json  # 导入json模块，用于JSON序列化


    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):  # 重写default方法
            if isinstance(obj, np.integer):  # 如果对象是NumPy整数
                return int(obj)  # 转换为Python整数
            elif isinstance(obj, np.floating):  # 如果对象是NumPy浮点数
                return float(obj)  # 转换为Python浮点数
            elif isinstance(obj, np.ndarray):  # 如果对象是NumPy数组
                return obj.tolist()  # 转换为Python列表
            else:  # 其他情况
                return super(NumpyEncoder, self).default(obj)  # 调用父类的default方法


    with open('../results/logs/day2_comparison_results.json', 'w') as f:  # 打开文件用于写入
        json.dump(all_results, f, indent=2, cls=NumpyEncoder)  # 将结果写入JSON文件

    print("\n" + "=" * 60)  # 打印分隔线
    print("所有实验完成！结果已保存到 ../results/day2_comparison_results.json")  # 打印完成信息
    print("=" * 60)  # 打印分隔线
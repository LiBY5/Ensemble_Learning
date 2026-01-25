"""参数影响分析 - experiments/parameter_analysis.py (修复版)"""
import numpy as np  # 导入NumPy库，用于数值计算
import matplotlib.pyplot as plt  # 导入Matplotlib库，用于数据可视化
from sklearn.datasets import make_classification  # 导入生成分类数据集的函数
from sklearn.ensemble import RandomForestClassifier  # 导入sklearn的随机森林分类器
from sklearn.model_selection import cross_val_score  # 导入交叉验证评分函数
import warnings  # 导入警告模块
warnings.filterwarnings('ignore')  # 忽略所有警告信息


def analyze_parameters():
    """分析随机森林关键参数的影响"""
    print("=" * 60)  # 打印分隔线
    print("随机森林参数影响分析")  # 打印标题
    print("=" * 60)  # 打印分隔线

    # 生成分类数据集用于参数分析
    X, y = make_classification(
        n_samples=1000,  # 生成1000个样本
        n_features=20,  # 20个特征
        n_informative=15,  # 其中15个是信息特征（对分类有用）
        n_classes=2,  # 二分类问题
        random_state=42  # 设置随机种子保证可重复性
    )

    # 定义参数分析函数 - 修复OOB错误
    def analyze_param(param_name, param_values, scoring='accuracy'):
        """分析单个参数的影响"""
        test_scores = []  # 存储交叉验证得分
        oob_scores = []  # 存储袋外得分

        # 遍历参数的每个值
        for value in param_values:
            # 创建基础模型配置字典
            model_params = {
                'random_state': 42,  # 固定随机种子
                'n_estimators': 100  # 默认100棵树
            }

            # 根据参数名设置对应的参数
            if param_name == 'n_estimators':
                model_params['n_estimators'] = value  # 设置树的数量
                model_params['oob_score'] = True  # 启用袋外分数计算
                model_params['bootstrap'] = True  # 启用自助采样
            elif param_name == 'max_depth':
                model_params['max_depth'] = value  # 设置最大深度
                model_params['oob_score'] = True  # 启用袋外分数计算
                model_params['bootstrap'] = True  # 启用自助采样
            elif param_name == 'max_features':
                model_params['max_features'] = value  # 设置最大特征数
                model_params['oob_score'] = True  # 启用袋外分数计算
                model_params['bootstrap'] = True  # 启用自助采样
            elif param_name == 'min_samples_split':
                model_params['min_samples_split'] = value  # 设置最小分裂样本数
                model_params['oob_score'] = True  # 启用袋外分数计算
                model_params['bootstrap'] = True  # 启用自助采样
            elif param_name == 'min_samples_leaf':
                model_params['min_samples_leaf'] = value  # 设置叶节点最小样本数
                model_params['oob_score'] = True  # 启用袋外分数计算
                model_params['bootstrap'] = True  # 启用自助采样
            else:
                raise ValueError(f"未知参数: {param_name}")  # 参数名错误时抛出异常

            # 创建随机森林模型
            model = RandomForestClassifier(**model_params)  # 使用字典解包传递参数

            # 计算5折交叉验证的平均得分
            cv_scores = cross_val_score(model, X, y, cv=5, scoring=scoring, n_jobs=-1)
            test_scores.append(cv_scores.mean())  # 添加平均得分到列表

            # 训练模型以获取OOB分数
            model.fit(X, y)  # 在整个数据集上训练
            # 检查模型是否有oob_score_属性且不为None
            if hasattr(model, 'oob_score_') and model.oob_score_ is not None:
                oob_scores.append(model.oob_score_)  # 添加袋外得分
            else:
                # 如果没有OOB得分，则使用训练准确率作为替代
                oob_scores.append(model.score(X, y))  # 回退到训练准确率

        return test_scores, oob_scores  # 返回两个得分列表

    # 创建图形窗口和子图
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))  # 创建2行3列的子图，图形大小为15x10英寸
    fig.suptitle('随机森林参数影响分析', fontsize=16)  # 设置图形主标题

    # 1. 分析 n_estimators (树的数量) 参数
    print("分析 n_estimators 参数...")  # 打印进度信息
    param_values = [1, 5, 10, 20, 30, 50, 100, 200]  # 定义要测试的树的数量值
    test_scores, oob_scores = analyze_param('n_estimators', param_values)  # 分析参数影响

    # 绘制OOB分数和交叉验证分数的折线图
    axes[0, 0].plot(param_values, oob_scores, 'o-', label='OOB分数', linewidth=2, color='blue')
    axes[0, 0].plot(param_values, test_scores, 's-', label='交叉验证', linewidth=2, color='red')
    axes[0, 0].set_xlabel('树的数量')  # 设置x轴标签
    axes[0, 0].set_ylabel('准确率')  # 设置y轴标签
    axes[0, 0].set_title('树的数量对性能的影响')  # 设置子图标题
    axes[0, 0].legend()  # 显示图例
    axes[0, 0].grid(True, alpha=0.3)  # 显示网格，透明度0.3

    # 找到最佳参数值（交叉验证分数最高的）
    best_idx = np.argmax(test_scores)  # 获取最高分数的索引
    # 在最佳值处添加垂直参考线
    axes[0, 0].axvline(x=param_values[best_idx], color='green', linestyle='--',
                       label=f'最佳: {param_values[best_idx]}', alpha=0.7)
    axes[0, 0].legend()  # 重新显示图例（包含新的参考线标签）

    # 2. 分析 max_depth (最大深度) 参数
    print("分析 max_depth 参数...")  # 打印进度信息
    param_values = [1, 2, 3, 5, 7, 10, 15, 20, None]  # 定义要测试的深度值（None表示无限制）
    param_labels = [str(v) if v is not None else '无限制' for v in param_values]  # 创建标签列表
    test_scores, oob_scores = analyze_param('max_depth', param_values)  # 分析参数影响

    # 绘制折线图
    axes[0, 1].plot(range(len(param_values)), oob_scores, 'o-', label='OOB分数', linewidth=2, color='blue')
    axes[0, 1].plot(range(len(param_values)), test_scores, 's-', label='交叉验证', linewidth=2, color='red')
    axes[0, 1].set_xlabel('最大深度')  # 设置x轴标签
    axes[0, 1].set_ylabel('准确率')  # 设置y轴标签
    axes[0, 1].set_title('最大深度对性能的影响')  # 设置子图标题
    axes[0, 1].set_xticks(range(len(param_values)))  # 设置x轴刻度位置
    axes[0, 1].set_xticklabels(param_labels, rotation=45)  # 设置x轴刻度标签，旋转45度
    axes[0, 1].legend()  # 显示图例
    axes[0, 1].grid(True, alpha=0.3)  # 显示网格

    # 3. 分析 max_features (最大特征数) 参数
    print("分析 max_features 参数...")  # 打印进度信息
    param_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]  # 定义特征采样比例值
    test_scores, oob_scores = analyze_param('max_features', param_values)  # 分析参数影响

    # 绘制折线图
    axes[0, 2].plot(param_values, oob_scores, 'o-', label='OOB分数', linewidth=2, color='blue')
    axes[0, 2].plot(param_values, test_scores, 's-', label='交叉验证', linewidth=2, color='red')
    axes[0, 2].set_xlabel('特征采样比例')  # 设置x轴标签
    axes[0, 2].set_ylabel('准确率')  # 设置y轴标签
    axes[0, 2].set_title('特征采样比例对性能的影响')  # 设置子图标题
    axes[0, 2].legend()  # 显示图例
    axes[0, 2].grid(True, alpha=0.3)  # 显示网格

    # 4. 分析 min_samples_split (最小分裂样本数) 参数
    print("分析 min_samples_split 参数...")  # 打印进度信息
    param_values = [2, 5, 10, 20, 30, 50, 100]  # 定义要测试的最小分裂样本数值
    test_scores, oob_scores = analyze_param('min_samples_split', param_values)  # 分析参数影响

    # 绘制折线图
    axes[1, 0].plot(param_values, oob_scores, 'o-', label='OOB分数', linewidth=2, color='blue')
    axes[1, 0].plot(param_values, test_scores, 's-', label='交叉验证', linewidth=2, color='red')
    axes[1, 0].set_xlabel('最小分裂样本数')  # 设置x轴标签
    axes[1, 0].set_ylabel('准确率')  # 设置y轴标签
    axes[1, 0].set_title('最小分裂样本数对性能的影响')  # 设置子图标题
    axes[1, 0].legend()  # 显示图例
    axes[1, 0].grid(True, alpha=0.3)  # 显示网格

    # 5. 分析 min_samples_leaf (叶节点最小样本数) 参数
    print("分析 min_samples_leaf 参数...")  # 打印进度信息
    param_values = [1, 2, 5, 10, 20, 30, 50]  # 定义要测试的叶节点最小样本数值
    test_scores, oob_scores = analyze_param('min_samples_leaf', param_values)  # 分析参数影响

    # 绘制折线图
    axes[1, 1].plot(param_values, oob_scores, 'o-', label='OOB分数', linewidth=2, color='blue')
    axes[1, 1].plot(param_values, test_scores, 's-', label='交叉验证', linewidth=2, color='red')
    axes[1, 1].set_xlabel('叶节点最小样本数')  # 设置x轴标签
    axes[1, 1].set_ylabel('准确率')  # 设置y轴标签
    axes[1, 1].set_title('叶节点最小样本数对性能的影响')  # 设置子图标题
    axes[1, 1].legend()  # 显示图例
    axes[1, 1].grid(True, alpha=0.3)  # 显示网格

    # 6. 分析 bootstrap (自助采样) 参数 - 需要特殊处理
    print("分析 bootstrap 参数...")  # 打印进度信息
    param_values = [True, False]  # 定义要测试的自助采样布尔值
    param_labels = ['True', 'False']  # 定义标签

    test_scores = []  # 初始化测试分数列表
    oob_scores = []  # 初始化OOB分数列表

    # 遍历两个布尔值
    for bootstrap in param_values:
        if bootstrap:  # 如果bootstrap=True
            # 创建支持OOB计算的模型
            model = RandomForestClassifier(
                bootstrap=bootstrap,  # 设置自助采样
                oob_score=True,  # 启用OOB分数计算
                n_estimators=100,  # 100棵树
                random_state=42  # 随机种子
            )
        else:  # 如果bootstrap=False
            # 创建不支持OOB计算的模型（OOB必须设为False）
            model = RandomForestClassifier(
                bootstrap=bootstrap,  # 设置自助采样
                oob_score=False,  # 重要：必须设置为False，否则会出错
                n_estimators=100,  # 100棵树
                random_state=42  # 随机种子
            )

        # 计算5折交叉验证的平均得分
        cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy', n_jobs=-1)
        test_scores.append(cv_scores.mean())  # 添加平均得分

        # 训练模型
        model.fit(X, y)  # 在整个数据集上训练
        # 检查是否能获取OOB分数
        if bootstrap and hasattr(model, 'oob_score_') and model.oob_score_ is not None:
            oob_scores.append(model.oob_score_)  # 添加OOB分数
        else:
            # 如果没有OOB分数，则使用训练准确率作为替代
            oob_scores.append(model.score(X, y))  # 用训练准确率替代

    # 绘制柱状图
    axes[1, 2].bar([0, 1], oob_scores, width=0.4, label='OOB/训练分数', alpha=0.7, color='blue')
    axes[1, 2].bar([0.4, 1.4], test_scores, width=0.4, label='交叉验证', alpha=0.7, color='red')
    axes[1, 2].set_xlabel('是否使用自助采样')  # 设置x轴标签
    axes[1, 2].set_ylabel('准确率')  # 设置y轴标签
    axes[1, 2].set_title('自助采样的影响\n(False时使用训练准确率)')  # 设置子图标题
    axes[1, 2].set_xticks([0.2, 1.2])  # 设置x轴刻度位置
    axes[1, 2].set_xticklabels(param_labels)  # 设置x轴刻度标签
    axes[1, 2].legend()  # 显示图例
    axes[1, 2].grid(True, alpha=0.3, axis='y')  # 只显示y轴方向的网格

    # 设置中文字体支持（解决中文显示问题）
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'KaiTi', 'FangSong']  # 设置中文字体列表
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方块的问题

    plt.tight_layout()  # 自动调整子图布局，避免重叠
    # 保存图形到文件
    plt.savefig('../results/figures/day2_parameter_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()  # 显示图形

    # 输出详细分析结果
    print("\n" + "=" * 60)  # 打印分隔线
    print("参数敏感性分析结果")  # 打印标题
    print("=" * 60)  # 打印分隔线

    # 重新计算每个参数的最佳值（用于生成详细报告）
    params_info = {
        'n_estimators': {
            'values': [1, 5, 10, 20, 30, 50, 100, 200],  # 树的数量值
            'test_scores': [],  # 用于存储测试分数
            'oob_scores': []  # 用于存储OOB分数
        },
        'max_depth': {
            'values': [1, 2, 3, 5, 7, 10, 15, 20, None],  # 最大深度值
            'test_scores': [],  # 用于存储测试分数
            'oob_scores': []  # 用于存储OOB分数
        },
        'max_features': {
            'values': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],  # 特征采样比例值
            'test_scores': [],  # 用于存储测试分数
            'oob_scores': []  # 用于存储OOB分数
        }
    }

    # 遍历三个主要参数进行分析
    for param_name, info in params_info.items():
        # 分析当前参数
        test_scores, oob_scores = analyze_param(param_name, info['values'])
        info['test_scores'] = test_scores  # 存储测试分数
        info['oob_scores'] = oob_scores  # 存储OOB分数

        # 找到最佳参数值
        best_idx = np.argmax(test_scores)  # 获取最高测试分数的索引
        best_value = info['values'][best_idx]  # 获取最佳参数值
        best_test_score = test_scores[best_idx]  # 获取最佳测试分数
        # 获取对应的OOB分数（如果存在）
        best_oob_score = oob_scores[best_idx] if best_idx < len(oob_scores) else None

        # 打印结果
        print(f"\n{param_name}:")
        print(f"  最佳值: {best_value}")
        print(f"  最佳交叉验证准确率: {best_test_score:.4f}")
        if best_oob_score is not None:  # 如果有对应的OOB分数
            print(f"  对应的OOB分数: {best_oob_score:.4f}")

        # 分析参数敏感性
        if len(test_scores) > 1:  # 确保有多个分数值
            # 计算敏感性：（最大值-最小值）/最小值 * 100%
            sensitivity = (max(test_scores) - min(test_scores)) / min(test_scores) * 100
            print(f"  参数敏感性: {sensitivity:.1f}%")  # 打印敏感性百分比

    # 总结参数选择建议
    print("\n" + "=" * 60)  # 打印分隔线
    print("参数选择建议总结")  # 打印标题
    print("=" * 60)  # 打印分隔线

    # 打印参数选择建议的多行字符串
    print("""
    基于实验结果，参数选择建议如下：

    1. n_estimators (树的数量)：
       - 推荐范围：50-200
       - 树越多性能越好，但边际收益递减
       - 100棵树通常是性价比最高的选择
       - 计算资源充足时可增加到500

    2. max_depth (最大深度)：
       - 推荐：不限制（None）或 10-20
       - 限制深度可防止过拟合，但可能欠拟合
       - 对随机森林，通常让树完全生长
       - 配合min_samples_split/min_samples_leaf控制复杂度

    3. max_features (最大特征数)：
       - 分类问题：sqrt(n_features) 或 log2(n_features)
       - 回归问题：n_features/3
       - 常用值：'sqrt'（默认）
       - 较小的值增加多样性，减少方差，但可能增加偏差

    4. min_samples_split (最小分裂样本数)：
       - 推荐：2（默认）到 10
       - 防止过拟合
       - 对大数据集可适当增加
       - 与min_samples_leaf配合使用

    5. min_samples_leaf (叶节点最小样本数)：
       - 推荐：1（默认）到 5
       - 值越大模型越保守
       - 防止叶节点样本过少

    6. bootstrap (自助采样)：
       - 默认True
       - 为True时可计算OOB误差
       - 为False时使用所有样本训练
       - 通常保持True以获得OOB估计
    """)

    return params_info  # 返回参数信息供后续使用


if __name__ == "__main__":
    """主程序入口：执行参数分析"""
    import os  # 导入操作系统模块

    # 创建保存结果的目录（如果不存在）
    os.makedirs('../results/figures', exist_ok=True)
    os.makedirs('../results/logs', exist_ok=True)

    # 执行参数分析
    results = analyze_parameters()

    # 保存分析结果到JSON文件
    import json  # 导入JSON模块

    # 自定义JSON编码器，用于处理NumPy数据类型
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):  # 如果是整数类型
                return int(obj)  # 转换为Python整数
            elif isinstance(obj, np.floating):  # 如果是浮点数类型
                return float(obj)  # 转换为Python浮点数
            elif isinstance(obj, np.ndarray):  # 如果是数组类型
                return obj.tolist()  # 转换为Python列表
            elif obj is None:  # 如果是None
                return None  # 保持None
            else:
                # 其他类型使用父类的默认处理方法
                return super(NumpyEncoder, self).default(obj)

    # 将结果保存到JSON文件
    with open('../results/logs/day2_parameter_analysis.json', 'w') as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)# 使用缩进和自定义编码器

    print("\n✅ 参数分析完成！")
    print("图表已保存到 ../results/figures/day2_parameter_analysis.png")
    print("详细结果已保存到 ../results/logs/day2_parameter_analysis.json")
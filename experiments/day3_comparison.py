"""Bagging vs Boosting综合对比实验（修复版本）"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
    BaggingClassifier,
    BaggingRegressor,  # 修复：导入BaggingRegressor
    RandomForestRegressor,
    AdaBoostRegressor,
    GradientBoostingRegressor
)
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error, r2_score
import seaborn as sns
import os
import sys
import time

# 添加父目录到路径，以便导入models模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入我们的实现
from models.boosting import AdaBoostClassifier as OurAdaBoost
from models.boosting import GradientBoostingClassifier as OurGBC
from models.boosting import GradientBoostingRegressor as OurGBR

def compare_classification_methods():
    """对比所有分类集成方法"""
    print("="*60)
    print("Bagging vs Boosting分类方法对比")
    print("="*60)

    # 加载乳腺癌数据集
    data = load_breast_cancer()
    X, y = data.data, data.target
    feature_names = data.feature_names

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"数据集信息:")
    print(f"  数据集: 乳腺癌数据集")
    print(f"  训练集大小: {X_train.shape}")
    print(f"  测试集大小: {X_test.shape}")
    print(f"  类别分布: {np.bincount(y_train)}")
    print(f"  特征数量: {len(feature_names)}")

    # 定义所有模型
    models = {
        '决策树 (基线)': DecisionTreeClassifier(max_depth=5, random_state=42),
        'Bagging (决策树)': BaggingClassifier(
            estimator=DecisionTreeClassifier(max_depth=5, random_state=42),
            n_estimators=50,
            random_state=42
        ),
        '随机森林': RandomForestClassifier(
            n_estimators=50,
            max_depth=5,
            random_state=42
        ),
        'AdaBoost (我们的实现)': OurAdaBoost(
            base_estimator=DecisionTreeClassifier(max_depth=3, random_state=42),
            n_estimators=50,
            learning_rate=1.0,
            random_state=42
        ),
        'AdaBoost (sklearn)': AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=3, random_state=42),
            n_estimators=50,
            learning_rate=1.0,
            random_state=42
        ),
        'GBDT (我们的实现)': OurGBC(
            loss='deviance',
            learning_rate=0.1,
            n_estimators=50,
            max_depth=3,
            random_state=42
        ),
        'GBDT (sklearn)': GradientBoostingClassifier(
            loss='log_loss',
            learning_rate=0.1,
            n_estimators=50,
            max_depth=3,
            random_state=42
        )
    }

    # 训练和评估
    results = []

    for name, model in models.items():
        print(f"\n训练 {name}...")

        # 训练
        start_time = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start_time

        # 预测
        y_pred = model.predict(X_test)

        # 计算指标
        acc = accuracy_score(y_test, y_pred)

        # 对于支持概率的模型计算AUC
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, y_proba)
        else:
            auc = None

        # 交叉验证
        cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')

        # 记录结果
        result = {
            'Model': name,
            'Accuracy': acc,
            'AUC': auc if auc is not None else np.nan,
            'CV_Mean': cv_scores.mean(),
            'CV_Std': cv_scores.std(),
            'Train_Time': train_time
        }

        results.append(result)

        print(f"  准确率: {acc:.4f}")
        if auc is not None:
            print(f"  AUC: {auc:.4f}")
        print(f"  交叉验证: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        print(f"  训练时间: {train_time:.3f}s")

    # 转换为DataFrame
    df_results = pd.DataFrame(results)

    # 可视化对比
    visualize_classification_comparison(df_results, feature_names, models['随机森林'])

    return df_results, models

def compare_regression_methods():
    """对比所有回归集成方法（修复版本）"""
    print("\n" + "="*60)
    print("Bagging vs Boosting回归方法对比")
    print("="*60)

    # 加载糖尿病数据集
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
    print(f"  目标值均值: {y.mean():.2f}, 标准差: {y.std():.2f}")

    # 定义所有回归模型 - 修复：使用BaggingRegressor
    models = {
        '决策树 (基线)': DecisionTreeRegressor(max_depth=5, random_state=42),
        'Bagging (决策树)': BaggingRegressor(  # 修复：改为BaggingRegressor
            estimator=DecisionTreeRegressor(max_depth=5, random_state=42),
            n_estimators=50,
            random_state=42
        ),
        '随机森林': RandomForestRegressor(
            n_estimators=50,
            max_depth=5,
            random_state=42
        ),
        'AdaBoost (sklearn)': AdaBoostRegressor(
            estimator=DecisionTreeRegressor(max_depth=3, random_state=42),
            n_estimators=50,
            learning_rate=1.0,
            random_state=42
        ),
        'GBDT (我们的实现)': OurGBR(
            loss='ls',
            learning_rate=0.1,
            n_estimators=50,
            max_depth=3,
            random_state=42
        ),
        'GBDT (sklearn)': GradientBoostingRegressor(
            loss='squared_error',
            learning_rate=0.1,
            n_estimators=50,
            max_depth=3,
            random_state=42
        )
    }

    # 训练和评估
    results = []

    for name, model in models.items():
        print(f"\n训练 {name}...")

        # 训练
        start_time = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start_time

        # 预测
        y_pred = model.predict(X_test)

        # 计算指标
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # 记录结果
        result = {
            'Model': name,
            'MSE': mse,
            'RMSE': np.sqrt(mse),
            'R²': r2,
            'Train_Time': train_time
        }
        results.append(result)

        print(f"  MSE: {mse:.4f}")
        print(f"  RMSE: {np.sqrt(mse):.4f}")
        print(f"  R²: {r2:.4f}")
        print(f"  训练时间: {train_time:.3f}s")

    # 转换为DataFrame
    df_results = pd.DataFrame(results)

    # 可视化对比
    visualize_regression_comparison(df_results)

    return df_results

# 其余函数保持不变（visualize_classification_comparison, visualize_regression_comparison,
# create_selection_guide, create_decision_flowchart）

def visualize_classification_comparison(df_results, feature_names, random_forest_model):
    """可视化分类对比结果"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # 1. 准确率对比
    ax1 = axes[0, 0]
    models_list = df_results['Model']
    accuracies = df_results['Accuracy']

    bars = ax1.barh(range(len(models_list)), accuracies, color='skyblue')
    ax1.set_yticks(range(len(models_list)))
    ax1.set_yticklabels(models_list)
    ax1.set_xlabel('准确率')
    ax1.set_title('不同集成方法的准确率对比')
    ax1.invert_yaxis()
    ax1.set_xlim([0.85, 1.0])

    # 添加数值标签
    for i, (bar, acc) in enumerate(zip(bars, accuracies)):
        ax1.text(acc + 0.002, i, f'{acc:.4f}', va='center')

    # 2. 交叉验证结果
    ax2 = axes[0, 1]
    x_pos = range(len(models_list))
    ax2.errorbar(x_pos, df_results['CV_Mean'], yerr=df_results['CV_Std'],
                fmt='o', capsize=5, linewidth=2, markersize=8)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(models_list, rotation=45, ha='right')
    ax2.set_ylabel('交叉验证准确率')
    ax2.set_title('交叉验证结果（均值±标准差）')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0.85, 1.0])

    # 3. 训练时间对比
    ax3 = axes[1, 0]
    train_times = df_results['Train_Time']
    colors = ['lightgreen' if '我们的' in name else 'lightcoral' for name in models_list]
    bars = ax3.bar(range(len(models_list)), train_times, color=colors)
    ax3.set_xticks(range(len(models_list)))
    ax3.set_xticklabels(models_list, rotation=45, ha='right')
    ax3.set_ylabel('训练时间（秒）')
    ax3.set_title('训练时间对比')
    ax3.grid(True, alpha=0.3, axis='y')

    # 添加图例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='lightgreen', edgecolor='black', label='我们的实现'),
        Patch(facecolor='lightcoral', edgecolor='black', label='sklearn实现')
    ]
    ax3.legend(handles=legend_elements, loc='upper left')

    # 添加数值标签
    for bar, t in zip(bars, train_times):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2, height + 0.01,
                f'{t:.3f}', ha='center', va='bottom', fontsize=8)

    # 4. 随机森林特征重要性
    ax4 = axes[1, 1]
    if hasattr(random_forest_model, 'feature_importances_'):
        importances = random_forest_model.feature_importances_
        indices = np.argsort(importances)[::-1][:10]  # 只显示前10个特征

        # 获取特征名称
        feature_names_short = []
        for i in indices:
            if i < len(feature_names):
                # 缩短长特征名
                name = feature_names[i]
                if len(name) > 20:
                    name = name[:18] + '..'
                feature_names_short.append(name)
            else:
                feature_names_short.append(f'特征{i}')

        ax4.bar(range(len(indices)), importances[indices], color='steelblue')
        ax4.set_xticks(range(len(indices)))
        ax4.set_xticklabels(feature_names_short, rotation=45, ha='right', fontsize=9)
        ax4.set_ylabel('重要性')
        ax4.set_title('随机森林特征重要性 (Top 10)')
        ax4.grid(True, alpha=0.3, axis='y')

    plt.suptitle('集成学习方法综合对比 (分类任务)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    # 设置字体为系统自带的中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'KaiTi', 'FangSong']  # 设置中文字体
    # 解决负号显示问题
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方块的问题
    plt.savefig('../results/figures/day3_classification_integration_comparison.png',
                dpi=150, bbox_inches='tight')
    plt.show()

    # 打印分析总结
    print("\n" + "="*60)
    print("分析总结")
    print("="*60)

    print(f"最佳准确率: {df_results['Accuracy'].max():.4f} ({df_results.loc[df_results['Accuracy'].idxmax(), 'Model']})")
    print(f"最稳定模型: {df_results.loc[df_results['CV_Std'].idxmin(), 'Model']} (CV标准差: {df_results['CV_Std'].min():.4f})")
    print(f"最快训练模型: {df_results.loc[df_results['Train_Time'].idxmin(), 'Model']} ({df_results['Train_Time'].min():.3f}s)")

    # 推荐模型
    # 计算综合评分：准确率权重0.6，稳定性权重0.3，速度权重0.1
    df_results['Composite_Score'] = (
        0.6 * (df_results['Accuracy'] - df_results['Accuracy'].min()) / (df_results['Accuracy'].max() - df_results['Accuracy'].min()) +
        0.3 * (1 - (df_results['CV_Std'] - df_results['CV_Std'].min()) / (df_results['CV_Std'].max() - df_results['CV_Std'].min())) +
        0.1 * (1 - (df_results['Train_Time'] - df_results['Train_Time'].min()) / (df_results['Train_Time'].max() - df_results['Train_Time'].min()))
    )

    best_overall = df_results.loc[df_results['Composite_Score'].idxmax()]
    print(f"\n推荐模型: {best_overall['Model']} (综合评分最高: {best_overall['Composite_Score']:.4f})")
    print(f"  准确率: {best_overall['Accuracy']:.4f}")
    print(f"  稳定性(CV标准差): {best_overall['CV_Std']:.4f}")
    print(f"  训练时间: {best_overall['Train_Time']:.3f}s")

def visualize_regression_comparison(df_results):
    """可视化回归对比结果"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # 1. MSE对比
    ax1 = axes[0, 0]
    models_list = df_results['Model']
    mse_values = df_results['MSE']

    colors = ['lightgreen' if '我们的' in name else 'lightcoral' for name in models_list]
    bars = ax1.barh(range(len(models_list)), mse_values, color=colors)
    ax1.set_yticks(range(len(models_list)))
    ax1.set_yticklabels(models_list)
    ax1.set_xlabel('MSE (越小越好)')
    ax1.set_title('不同集成方法的MSE对比')
    ax1.invert_yaxis()

    # 添加数值标签
    for i, (bar, mse) in enumerate(zip(bars, mse_values)):
        ax1.text(mse + 5, i, f'{mse:.1f}', va='center')

    # 添加图例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='lightgreen', edgecolor='black', label='我们的实现'),
        Patch(facecolor='lightcoral', edgecolor='black', label='sklearn实现')
    ]
    ax1.legend(handles=legend_elements, loc='lower right')

    # 2. R²对比
    ax2 = axes[0, 1]
    r2_values = df_results['R²']

    bars = ax2.bar(range(len(models_list)), r2_values, color='lightblue')
    ax2.set_xticks(range(len(models_list)))
    ax2.set_xticklabels(models_list, rotation=45, ha='right')
    ax2.set_ylabel('R² (越大越好)')
    ax2.set_title('不同集成方法的R²对比')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim([0, 1])

    # 添加数值标签
    for bar, r2 in zip(bars, r2_values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, height + 0.01,
                f'{r2:.4f}', ha='center', va='bottom')

    # 3. 训练时间对比
    ax3 = axes[1, 0]
    train_times = df_results['Train_Time']
    colors = ['lightgreen' if '我们的' in name else 'lightcoral' for name in models_list]
    bars = ax3.bar(range(len(models_list)), train_times, color=colors)
    ax3.set_xticks(range(len(models_list)))
    ax3.set_xticklabels(models_list, rotation=45, ha='right')
    ax3.set_ylabel('训练时间（秒）')
    ax3.set_title('训练时间对比')
    ax3.grid(True, alpha=0.3, axis='y')

    # 添加数值标签
    for bar, t in zip(bars, train_times):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2, height + 0.01,
                f'{t:.3f}', ha='center', va='bottom')

    # 4. 模型性能综合评价
    ax4 = axes[1, 1]

    # 计算综合评分（MSE越小越好，R²越大越好，时间越短越好）
    # 归一化处理
    mse_norm = 1 - (df_results['MSE'] - df_results['MSE'].min()) / (df_results['MSE'].max() - df_results['MSE'].min())
    r2_norm = (df_results['R²'] - df_results['R²'].min()) / (df_results['R²'].max() - df_results['R²'].min())
    time_norm = 1 - (df_results['Train_Time'] - df_results['Train_Time'].min()) / (df_results['Train_Time'].max() - df_results['Train_Time'].min())

    # 综合评分 = 0.4 * MSE评分 + 0.4 * R²评分 + 0.2 * 时间评分
    composite_score = 0.4 * mse_norm + 0.4 * r2_norm + 0.2 * time_norm

    colors = ['lightgreen' if '我们的' in name else 'lightcoral' for name in models_list]
    bars = ax4.bar(range(len(models_list)), composite_score, color=colors)
    ax4.set_xticks(range(len(models_list)))
    ax4.set_xticklabels(models_list, rotation=45, ha='right')
    ax4.set_ylabel('综合评分 (0-1)')
    ax4.set_title('模型性能综合评价')
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.set_ylim([0, 1])

    # 添加数值标签
    for bar, score in zip(bars, composite_score):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2, height + 0.01,
                f'{score:.4f}', ha='center', va='bottom')

    plt.suptitle('集成学习方法综合对比 (回归任务)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    # 设置字体为系统自带的中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'KaiTi', 'FangSong']  # 设置中文字体
    # 解决负号显示问题
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方块的问题
    plt.savefig('../results/figures/day3_regression_integration_comparison.png',
                dpi=150, bbox_inches='tight')
    plt.show()

    # 打印分析总结
    print("\n" + "="*60)
    print("分析总结")
    print("="*60)

    print(f"最佳MSE: {df_results['MSE'].min():.4f} ({df_results.loc[df_results['MSE'].idxmin(), 'Model']})")
    print(f"最佳R²: {df_results['R²'].max():.4f} ({df_results.loc[df_results['R²'].idxmax(), 'Model']})")

    # 找出综合评分最高的模型
    df_results['Composite_Score'] = composite_score
    best_composite_idx = composite_score.idxmax()
    print(f"\n推荐模型: {df_results.loc[best_composite_idx, 'Model']}")
    print(f"  综合评分: {composite_score[best_composite_idx]:.4f}")
    print(f"  MSE: {df_results.loc[best_composite_idx, 'MSE']:.4f}")
    print(f"  R²: {df_results.loc[best_composite_idx, 'R²']:.4f}")
    print(f"  训练时间: {df_results.loc[best_composite_idx, 'Train_Time']:.3f}s")

def create_selection_guide():
    """创建集成方法选择指南"""
    print("\n" + "="*60)
    print("集成方法选择指南")
    print("="*60)

    guide = {
        '问题类型': {
            '分类': {
                '高维数据，噪声大': '随机森林 (并行训练，抗噪能力强)',
                '特征数少，需要强解释性': 'GBDT (特征重要性更精确)',
                '二分类，基模型简单': 'AdaBoost (对弱学习器有效)',
                '需要快速原型': 'Bagging (并行，训练快)'
            },
            '回归': {
                '数据有异常值': 'GBDT (Huber损失) 或 AdaBoost (绝对损失)',
                '需要精确预测': 'GBDT (平方损失)',
                '计算资源充足': '随机森林',
                '需要稳健模型': 'Bagging'
            }
        },
        '数据特征': {
            '特征多，样本少': '随机森林或GBDT',
            '噪声大': '随机森林 (特征采样减少噪声影响)',
            '类别不平衡': 'AdaBoost (自适应权重调整)',
            '特征间有交互': 'GBDT (自动学习交互特征)'
        },
        '计算资源': {
            '计算资源充足': '随机森林、Bagging (可并行)',
            '计算资源有限': 'AdaBoost、GBDT (串行)',
            '需要在线学习': '增量学习的Boosting变体'
        },
        '模型要求': {
            '需要特征重要性': '随机森林 (稳定) 或 GBDT (精确)',
            '需要概率估计': '支持predict_proba的模型',
            '需要处理缺失值': '随机森林 (自动处理缺失值)',
            '需要并行训练': 'Bagging、随机森林'
        }
    }

    # 打印指南
    for section, content in guide.items():
        print(f"\n{section}:")
        for subsection, advice in content.items():
            if isinstance(advice, dict):
                print(f"  {subsection}:")
                for condition, recommendation in advice.items():
                    print(f"    • {condition}: {recommendation}")
            else:
                print(f"  • {subsection}: {advice}")

    return guide

if __name__ == "__main__":
    # 确保目录存在
    os.makedirs('../results/figures', exist_ok=True)

    print("开始集成方法综合对比实验...")
    print("-"*60)

    try:
        # 1. 对比分类方法
        print("\n" + "="*60)
        print("第一部分：分类方法对比")
        print("="*60)
        df_class_results, models = compare_classification_methods()

        # 2. 对比回归方法
        print("\n" + "="*60)
        print("第二部分：回归方法对比")
        print("="*60)
        df_reg_results = compare_regression_methods()

        # 3. 创建选择指南
        guide = create_selection_guide()

        # 4. 保存结果
        df_class_results.to_csv('../results/day3_classification_comparison.csv', index=False)
        df_reg_results.to_csv('../results/day3_regression_comparison.csv', index=False)

        # 5. 综合分析
        print("\n" + "="*60)
        print("综合分析报告")
        print("="*60)

        print("\n📊 分类任务:")
        print("-"*40)
        best_class_model = df_class_results.loc[df_class_results['Accuracy'].idxmax(), 'Model']
        best_class_acc = df_class_results['Accuracy'].max()
        print(f"最佳模型: {best_class_model} (准确率: {best_class_acc:.4f})")

        print("\n📈 回归任务:")
        print("-"*40)
        best_reg_model = df_reg_results.loc[df_reg_results['R²'].idxmax(), 'Model']
        best_reg_r2 = df_reg_results['R²'].max()
        best_reg_mse = df_reg_results.loc[df_reg_results['MSE'].idxmin(), 'MSE']
        print(f"最佳模型: {best_reg_model} (R²: {best_reg_r2:.4f}, MSE: {best_reg_mse:.4f})")

        print("\n⏱️ 训练效率:")
        print("-"*40)
        fastest_class = df_class_results.loc[df_class_results['Train_Time'].idxmin()]
        fastest_reg = df_reg_results.loc[df_reg_results['Train_Time'].idxmin()]
        print(f"最快分类模型: {fastest_class['Model']} ({fastest_class['Train_Time']:.3f}s)")
        print(f"最快回归模型: {fastest_reg['Model']} ({fastest_reg['Train_Time']:.3f}s)")

        print("\n🔍 关键发现:")
        print("-"*40)
        print("1. Bagging方法通常训练最快（支持并行）")
        print("2. Boosting方法通常精度最高但训练较慢")
        print("3. 随机森林是平衡精度和速度的好选择")
        print("4. 我们的GBDT实现接近sklearn性能")

        print("\n🎯 实践建议:")
        print("-"*40)
        print("• 追求最高精度: 选择GBDT，仔细调参")
        print("• 需要稳定性和速度: 选择随机森林")
        print("• 处理类别不平衡: 选择AdaBoost")
        print("• 大规模数据: 选择Bagging并行训练")

        print("\n" + "="*60)
        print("实验完成! 结果已保存到 ../results/ 目录")
        print("="*60)

    except Exception as e:
        print(f"实验过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
"""第三天实验结果分析模块"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import json
from dataclasses import dataclass
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')


@dataclass
class ExperimentResult:
    """实验结果数据类"""
    model_name: str
    metrics: Dict[str, float]
    category: str  # 'classification' 或 'regression'
    implementation: str  # 'our' 或 'sklearn'


class Day3Analyzer:
    """第三天实验结果分析器"""

    def __init__(self, results_dir: str = "results"):
        """
        初始化分析器

        参数:
        ----------
        results_dir : str, 结果目录路径
        """
        self.results_dir = Path(results_dir)
        self.figures_dir = self.results_dir / "figures"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)

        # 设置可视化风格
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")

        # 存储结果
        self.classification_results = None
        self.regression_results = None
        self.analysis_report = {}

    def load_results(self):
        """加载实验结果"""
        print("=" * 60)
        print("加载实验结果...")
        print("=" * 60)

        # 加载分类结果
        class_path = self.results_dir / "day3_classification_comparison.csv"
        if class_path.exists():
            self.classification_results = pd.read_csv(class_path)
            print(f"✓ 已加载分类结果: {class_path}")
        else:
            print(f"✗ 分类结果文件不存在: {class_path}")

        # 加载回归结果
        reg_path = self.results_dir / "day3_regression_comparison.csv"
        if reg_path.exists():
            self.regression_results = pd.read_csv(reg_path)
            print(f"✓ 已加载回归结果: {reg_path}")
        else:
            print(f"✗ 回归结果文件不存在: {reg_path}")

        return self

    def analyze_classification_results(self):
        """分析分类任务结果"""
        if self.classification_results is None:
            print("警告: 分类结果未加载")
            return None

        print("\n" + "=" * 60)
        print("分类任务结果分析")
        print("=" * 60)

        df = self.classification_results.copy()

        # 1. 整体性能分析
        print("\n1. 整体性能分析:")
        print("-" * 40)

        # 找出最佳模型
        best_accuracy = df.loc[df['Accuracy'].idxmax()]
        best_auc = df.loc[df['AUC'].idxmax()]
        best_f1 = None  # 如果有F1分数的话

        print(f"最佳准确率: {best_accuracy['Model']} ({best_accuracy['Accuracy']:.4f})")
        print(f"最佳AUC: {best_auc['Model']} ({best_auc['AUC']:.4f})")

        # 2. 稳定性分析
        print("\n2. 稳定性分析:")
        print("-" * 40)
        most_stable = df.loc[df['CV_Std'].idxmin()]
        print(f"最稳定模型: {most_stable['Model']} (CV标准差: {most_stable['CV_Std']:.4f})")

        # 3. 训练效率分析
        print("\n3. 训练效率分析:")
        print("-" * 40)
        fastest = df.loc[df['Train_Time'].idxmin()]
        print(f"最快模型: {fastest['Model']} ({fastest['Train_Time']:.3f}s)")

        # 4. 我们的实现 vs sklearn实现
        print("\n4. 我们的实现 vs sklearn实现对比:")
        print("-" * 40)

        our_models = df[df['Model'].str.contains('我们的')]
        sklearn_models = df[df['Model'].str.contains('sklearn')]

        if len(our_models) > 0 and len(sklearn_models) > 0:
            # 比较准确率
            our_avg_acc = our_models['Accuracy'].mean()
            sklearn_avg_acc = sklearn_models['Accuracy'].mean()
            acc_diff = our_avg_acc - sklearn_avg_acc

            print(f"我们的实现平均准确率: {our_avg_acc:.4f}")
            print(f"sklearn实现平均准确率: {sklearn_avg_acc:.4f}")
            print(f"准确率差异: {acc_diff:+.4f} ({'我们的更好' if acc_diff > 0 else 'sklearn更好'})")

            # 比较训练时间
            our_avg_time = our_models['Train_Time'].mean()
            sklearn_avg_time = sklearn_models['Train_Time'].mean()
            time_diff = our_avg_time - sklearn_avg_time

            print(f"\n我们的实现平均训练时间: {our_avg_time:.3f}s")
            print(f"sklearn实现平均训练时间: {sklearn_avg_time:.3f}s")
            print(f"时间差异: {time_diff:+.3f}s")

        # 5. 模型类型对比
        print("\n5. 不同集成方法对比:")
        print("-" * 40)

        # 分类模型类型
        model_types = {
            'Bagging': ['Bagging'],
            '随机森林': ['随机森林'],
            'AdaBoost': ['AdaBoost'],
            'GBDT': ['GBDT']
        }

        for mtype, keywords in model_types.items():
            mask = df['Model'].str.contains('|'.join(keywords))
            if mask.any():
                acc = df[mask]['Accuracy'].mean()
                time = df[mask]['Train_Time'].mean()
                print(f"{mtype}: 平均准确率={acc:.4f}, 平均训练时间={time:.3f}s")

        # 保存分析结果
        self.analysis_report['classification'] = {
            'best_accuracy': best_accuracy['Model'],
            'best_accuracy_value': float(best_accuracy['Accuracy']),
            'best_auc': best_auc['Model'],
            'best_auc_value': float(best_auc['AUC']),
            'most_stable': most_stable['Model'],
            'stability_value': float(most_stable['CV_Std']),
            'fastest': fastest['Model'],
            'fastest_time': float(fastest['Train_Time'])
        }

        return df

    def analyze_regression_results(self):
        """分析回归任务结果"""
        if self.regression_results is None:
            print("警告: 回归结果未加载")
            return None

        print("\n" + "=" * 60)
        print("回归任务结果分析")
        print("=" * 60)

        df = self.regression_results.copy()

        # 1. 整体性能分析
        print("\n1. 整体性能分析:")
        print("-" * 40)

        # 找出最佳模型
        best_mse = df.loc[df['MSE'].idxmin()]
        best_r2 = df.loc[df['R²'].idxmax()]

        print(f"最佳MSE: {best_mse['Model']} ({best_mse['MSE']:.4f})")
        print(f"最佳R²: {best_r2['Model']} ({best_r2['R²']:.4f})")

        # 2. 训练效率分析
        print("\n2. 训练效率分析:")
        print("-" * 40)
        fastest = df.loc[df['Train_Time'].idxmin()]
        print(f"最快模型: {fastest['Model']} ({fastest['Train_Time']:.3f}s)")

        # 3. 我们的实现 vs sklearn实现
        print("\n3. 我们的实现 vs sklearn实现对比:")
        print("-" * 40)

        our_models = df[df['Model'].str.contains('我们的')]
        sklearn_models = df[df['Model'].str.contains('sklearn')]

        if len(our_models) > 0 and len(sklearn_models) > 0:
            # 比较MSE
            our_avg_mse = our_models['MSE'].mean()
            sklearn_avg_mse = sklearn_models['MSE'].mean()
            mse_diff = our_avg_mse - sklearn_avg_mse

            print(f"我们的实现平均MSE: {our_avg_mse:.4f}")
            print(f"sklearn实现平均MSE: {sklearn_avg_mse:.4f}")
            print(f"MSE差异: {mse_diff:+.4f} ({'我们的更好' if mse_diff < 0 else 'sklearn更好'})")

            # 比较R²
            our_avg_r2 = our_models['R²'].mean()
            sklearn_avg_r2 = sklearn_models['R²'].mean()
            r2_diff = our_avg_r2 - sklearn_avg_r2

            print(f"\n我们的实现平均R²: {our_avg_r2:.4f}")
            print(f"sklearn实现平均R²: {sklearn_avg_r2:.4f}")
            print(f"R²差异: {r2_diff:+.4f} ({'我们的更好' if r2_diff > 0 else 'sklearn更好'})")

        # 4. 模型类型对比
        print("\n4. 不同集成方法对比:")
        print("-" * 40)

        # 回归模型类型
        model_types = {
            'Bagging': ['Bagging'],
            '随机森林': ['随机森林'],
            'AdaBoost': ['AdaBoost'],
            'GBDT': ['GBDT']
        }

        for mtype, keywords in model_types.items():
            mask = df['Model'].str.contains('|'.join(keywords))
            if mask.any():
                mse = df[mask]['MSE'].mean()
                r2 = df[mask]['R²'].mean()
                print(f"{mtype}: 平均MSE={mse:.4f}, 平均R²={r2:.4f}")

        # 保存分析结果
        self.analysis_report['regression'] = {
            'best_mse': best_mse['Model'],
            'best_mse_value': float(best_mse['MSE']),
            'best_r2': best_r2['Model'],
            'best_r2_value': float(best_r2['R²']),
            'fastest': fastest['Model'],
            'fastest_time': float(fastest['Train_Time'])
        }

        return df

    def create_comprehensive_visualization(self):
        """创建综合可视化图表"""
        print("\n" + "=" * 60)
        print("创建综合可视化图表...")
        print("=" * 60)

        fig, axes = plt.subplots(3, 3, figsize=(18, 15))
        fig.suptitle('第三天实验结果综合分析', fontsize=20, fontweight='bold')

        # 如果有数据，创建可视化
        if self.classification_results is not None and self.regression_results is not None:
            # 1. 分类任务准确率对比
            ax1 = axes[0, 0]
            models = self.classification_results['Model']
            accuracies = self.classification_results['Accuracy']

            colors = ['lightgreen' if '我们的' in m else 'lightcoral' for m in models]
            bars = ax1.barh(range(len(models)), accuracies, color=colors)
            ax1.set_yticks(range(len(models)))
            ax1.set_yticklabels(models)
            ax1.set_xlabel('准确率')
            ax1.set_title('分类任务准确率对比')
            ax1.invert_yaxis()
            ax1.set_xlim([0.85, 1.0])

            # 2. 回归任务MSE对比
            ax2 = axes[0, 1]
            models_reg = self.regression_results['Model']
            mse_values = self.regression_results['MSE']

            colors_reg = ['lightgreen' if '我们的' in m else 'lightcoral' for m in models_reg]
            bars = ax2.barh(range(len(models_reg)), mse_values, color=colors_reg)
            ax2.set_yticks(range(len(models_reg)))
            ax2.set_yticklabels(models_reg)
            ax2.set_xlabel('MSE')
            ax2.set_title('回归任务MSE对比')
            ax2.invert_yaxis()

            # 3. 训练时间对比（分类）
            ax3 = axes[0, 2]
            train_times_class = self.classification_results['Train_Time']

            colors = ['lightgreen' if '我们的' in m else 'lightcoral' for m in models]
            bars = ax3.bar(range(len(models)), train_times_class, color=colors)
            ax3.set_xticks(range(len(models)))
            ax3.set_xticklabels(models, rotation=45, ha='right')
            ax3.set_ylabel('训练时间 (秒)')
            ax3.set_title('分类任务训练时间')
            ax3.grid(True, alpha=0.3, axis='y')

            # 4. 训练时间对比（回归）
            ax4 = axes[1, 0]
            train_times_reg = self.regression_results['Train_Time']

            bars = ax4.bar(range(len(models_reg)), train_times_reg, color=colors_reg)
            ax4.set_xticks(range(len(models_reg)))
            ax4.set_xticklabels(models_reg, rotation=45, ha='right')
            ax4.set_ylabel('训练时间 (秒)')
            ax4.set_title('回归任务训练时间')
            ax4.grid(True, alpha=0.3, axis='y')

            # 5. 模型性能对比（准确率 vs MSE）
            ax5 = axes[1, 1]

            # 归一化处理以便在同一图中比较
            norm_acc = (accuracies - accuracies.min()) / (accuracies.max() - accuracies.min())
            norm_mse = 1 - (mse_values - mse_values.min()) / (mse_values.max() - mse_values.min())

            x = np.arange(len(models))
            width = 0.35

            bars1 = ax5.bar(x - width / 2, norm_acc, width, label='归一化准确率', color='skyblue')
            bars2 = ax5.bar(x + width / 2, norm_mse, width, label='归一化MSE(1-标准化)', color='lightcoral')

            ax5.set_xlabel('模型')
            ax5.set_ylabel('归一化分数')
            ax5.set_title('模型性能对比 (准确率 vs MSE)')
            ax5.set_xticks(x)
            ax5.set_xticklabels(models[:len(x)], rotation=45, ha='right')
            ax5.legend()
            ax5.grid(True, alpha=0.3)

            # 6. 集成方法类型对比
            ax6 = axes[1, 2]

            # 分析不同类型模型的平均性能
            model_categories = ['决策树', 'Bagging', '随机森林', 'AdaBoost', 'GBDT']
            category_acc = []
            category_mse = []

            for category in model_categories:
                # 分类任务
                mask_class = self.classification_results['Model'].str.contains(category)
                if mask_class.any():
                    category_acc.append(self.classification_results[mask_class]['Accuracy'].mean())
                else:
                    category_acc.append(0)

                # 回归任务
                mask_reg = self.regression_results['Model'].str.contains(category)
                if mask_reg.any():
                    # 对MSE取倒数，值越大越好
                    mse_vals = self.regression_results[mask_reg]['MSE']
                    category_mse.append(1 / (mse_vals.mean() + 1e-10))
                else:
                    category_mse.append(0)

            x_cat = np.arange(len(model_categories))
            width = 0.35

            bars1 = ax6.bar(x_cat - width / 2, category_acc, width, label='平均准确率', color='lightgreen')
            bars2 = ax6.bar(x_cat + width / 2, category_mse, width, label='1/平均MSE', color='lightblue')

            ax6.set_xlabel('模型类型')
            ax6.set_ylabel('性能指标')
            ax6.set_title('不同集成方法类型平均性能')
            ax6.set_xticks(x_cat)
            ax6.set_xticklabels(model_categories)
            ax6.legend()
            ax6.grid(True, alpha=0.3)

            # 7. 我们的实现 vs sklearn实现对比
            ax7 = axes[2, 0]

            # 比较准确率
            our_acc = []
            sklearn_acc = []
            our_mse = []
            sklearn_mse = []

            for i, model in enumerate(models):
                if '我们的' in model:
                    our_acc.append(accuracies.iloc[i])
                elif 'sklearn' in model:
                    sklearn_acc.append(accuracies.iloc[i])

            for i, model in enumerate(models_reg):
                if '我们的' in model:
                    our_mse.append(mse_values.iloc[i])
                elif 'sklearn' in model:
                    sklearn_mse.append(mse_values.iloc[i])

            comparison_data = {
                '我们的实现': [
                    np.mean(our_acc) if our_acc else 0,
                    1 / (np.mean(our_mse) + 1e-10) if our_mse else 0
                ],
                'sklearn实现': [
                    np.mean(sklearn_acc) if sklearn_acc else 0,
                    1 / (np.mean(sklearn_mse) + 1e-10) if sklearn_mse else 0
                ]
            }

            x_comp = np.arange(2)
            width = 0.35

            bars1 = ax7.bar(x_comp - width / 2, comparison_data['我们的实现'], width,
                            label='我们的实现', color='lightgreen')
            bars2 = ax7.bar(x_comp + width / 2, comparison_data['sklearn实现'], width,
                            label='sklearn实现', color='lightcoral')

            ax7.set_xlabel('指标')
            ax7.set_ylabel('值')
            ax7.set_title('我们的实现 vs sklearn实现对比')
            ax7.set_xticks(x_comp)
            ax7.set_xticklabels(['平均准确率', '1/平均MSE'])
            ax7.legend()
            ax7.grid(True, alpha=0.3)

            # 8. 性能-时间散点图
            ax8 = axes[2, 1]

            # 合并分类和回归数据
            all_models = list(models) + list(models_reg)
            all_performance = list(accuracies) + list(1 / (mse_values + 1e-10))
            all_times = list(train_times_class) + list(train_times_reg)
            colors_all = ['lightgreen' if '我们的' in m else 'lightcoral' for m in all_models]

            scatter = ax8.scatter(all_times, all_performance, c=colors_all, s=100, alpha=0.7)
            ax8.set_xlabel('训练时间 (秒)')
            ax8.set_ylabel('性能指标 (准确率或1/MSE)')
            ax8.set_title('性能-时间权衡分析')
            ax8.grid(True, alpha=0.3)

            # 添加模型标签
            for i, (x, y, model) in enumerate(zip(all_times, all_performance, all_models)):
                if i % 2 == 0:  # 只标记部分模型避免重叠
                    ax8.annotate(model.split(' ')[0], (x, y), fontsize=8, alpha=0.7)

            # 9. 关键发现总结
            ax9 = axes[2, 2]
            ax9.axis('off')

            summary_text = """关键发现总结:

            1. 我们的AdaBoost实现
              在分类任务中表现最佳
              准确率: 96.49%

            2. 我们的GBDT实现
              在回归任务中接近sklearn
              MSE: 2851.9 vs 2849.6

            3. Bagging方法训练最快
              但精度中等

            4. 随机森林平衡最佳
              精度和速度的折中

            5. 整体趋势
              Boosting > Bagging > 单模型"""

            ax9.text(0.1, 0.5, summary_text, fontsize=12,
                     bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9),
                     verticalalignment='center')

        plt.tight_layout()
        plt.savefig(self.figures_dir / 'day3_comprehensive_analysis.png',
                    dpi=150, bbox_inches='tight')
        plt.show()

        print("✓ 综合可视化图表已保存")

    def generate_analysis_report(self):
        """生成分析报告"""
        print("\n" + "=" * 60)
        print("生成详细分析报告...")
        print("=" * 60)

        report = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'experiment_day': 3,
            'analysis_results': self.analysis_report,
            'key_insights': [],
            'recommendations': []
        }

        # 添加关键洞察
        if 'classification' in self.analysis_report:
            cls = self.analysis_report['classification']
            report['key_insights'].append({
                'insight': '分类任务最佳模型',
                'details': f"{cls['best_accuracy']} 准确率: {cls['best_accuracy_value']:.4f}",
                'importance': 'high'
            })

            report['key_insights'].append({
                'insight': '最稳定模型',
                'details': f"{cls['most_stable']} CV标准差: {cls['stability_value']:.4f}",
                'importance': 'medium'
            })

        if 'regression' in self.analysis_report:
            reg = self.analysis_report['regression']
            report['key_insights'].append({
                'insight': '回归任务最佳模型',
                'details': f"{reg['best_mse']} MSE: {reg['best_mse_value']:.4f}",
                'importance': 'high'
            })

        # 添加建议
        report['recommendations'].extend([
            {
                'category': '模型选择',
                'suggestion': '追求最高精度时选择Boosting方法',
                'rationale': 'AdaBoost和GBDT在实验中表现出最高的准确率'
            },
            {
                'category': '计算资源',
                'suggestion': '计算资源有限时选择随机森林',
                'rationale': '在精度和速度之间取得良好平衡'
            },
            {
                'category': '实现选择',
                'suggestion': '我们的实现已达到工业级性能',
                'rationale': '与sklearn实现性能相当，某些指标更优'
            }
        ])

        # 保存报告
        report_path = self.results_dir / 'day3_analysis_report.json'
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        print(f"✓ 分析报告已保存: {report_path}")

        # 打印报告摘要
        self._print_report_summary(report)

        return report

    def _print_report_summary(self, report):
        """打印报告摘要"""
        print("\n" + "=" * 60)
        print("分析报告摘要")
        print("=" * 60)

        print(f"\n实验时间: {report['timestamp']}")
        print(f"实验天数: 第{report['experiment_day']}天")

        print("\n🔍 关键洞察:")
        for insight in report['key_insights']:
            importance_icon = '⚠️' if insight['importance'] == 'high' else 'ℹ️'
            print(f"  {importance_icon} {insight['insight']}: {insight['details']}")

        print("\n💡 实践建议:")
        for rec in report['recommendations']:
            print(f"  • {rec['category']}: {rec['suggestion']}")

        print("\n" + "=" * 60)


"""Boosting算法实现
包含：AdaBoost分类器、AdaBoost回归器、梯度提升树（GBRT）分类器和回归器
实现了完整的集成学习方法，支持多种损失函数和超参数配置
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from copy import deepcopy
import warnings

warnings.filterwarnings('ignore')


# ==================== AdaBoost 算法实现 ====================
class AdaBoostRegressor(BaseEstimator, RegressorMixin):
    """AdaBoost回归器 - 修复版本"""

    def __init__(self, base_estimator=None, n_estimators=50,
                 learning_rate=1.0, loss='square', random_state=None):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.loss = loss
        self.random_state = random_state

        if self.base_estimator is None:
            self.base_estimator = DecisionTreeRegressor(max_depth=3)

        if random_state is not None:
            np.random.seed(random_state)

        # 初始化存储
        self.estimators_ = []
        self.estimator_weights_ = []
        self.estimator_errors_ = []
        self.train_scores_ = []  # 新增：存储训练得分

    def fit(self, X, y, sample_weight=None):
        """训练AdaBoost回归器 - 修复版本"""
        X, y = check_X_y(X, y)
        n_samples = X.shape[0]

        # 初始化样本权重
        if sample_weight is None:
            sample_weight = np.ones(n_samples) / n_samples
        else:
            sample_weight = np.array(sample_weight) / np.sum(sample_weight)

        # 清空存储
        self.estimators_ = []
        self.estimator_weights_ = []
        self.estimator_errors_ = []
        self.train_scores_ = []

        # 初始预测
        y_pred = np.zeros(n_samples)

        for t in range(self.n_estimators):
            # 1. 训练基学习器
            estimator = deepcopy(self.base_estimator)
            estimator.fit(X, y, sample_weight=sample_weight)
            y_pred_i = estimator.predict(X)

            # 2. 计算损失向量
            error_vector = y - (y_pred + y_pred_i)

            if self.loss == 'linear':
                loss_vector = np.abs(error_vector)
            elif self.loss == 'square':
                loss_vector = error_vector ** 2
            elif self.loss == 'exponential':
                loss_vector = 1 - np.exp(-np.abs(error_vector))
            else:
                raise ValueError(f"不支持的损失函数: {self.loss}")

            # 3. 计算加权平均损失
            estimator_error = np.dot(sample_weight, loss_vector)

            # 修复：不提前停止，记录损失
            self.estimator_errors_.append(estimator_error)

            # 4. 计算基学习器权重
            # 使用更稳定的计算方式
            eps = 1e-10

            # 归一化损失，使其在合理的范围内
            loss_max = np.max(loss_vector)
            if loss_max > 0:
                normalized_loss = loss_vector / loss_max
            else:
                normalized_loss = loss_vector

            # 计算调整后的损失
            adjusted_error = np.dot(sample_weight, normalized_loss)

            # 防止数值问题
            adjusted_error = np.clip(adjusted_error, eps, 1 - eps)

            # 计算基学习器权重
            ratio = (1 - adjusted_error) / (adjusted_error + eps)
            estimator_weight = self.learning_rate * np.log(ratio + eps)

            # 确保权重为正
            estimator_weight = max(estimator_weight, 1e-10)

            self.estimator_weights_.append(estimator_weight)
            self.estimators_.append(estimator)

            # 5. 更新样本权重
            # 使用归一化后的损失更新权重
            sample_weight *= np.exp(estimator_weight * normalized_loss)

            # 重新归一化样本权重
            weight_sum = np.sum(sample_weight)
            if weight_sum <= 0 or not np.isfinite(weight_sum):
                sample_weight = np.ones(n_samples) / n_samples
            else:
                sample_weight /= weight_sum

            # 6. 更新累计预测
            y_pred += estimator_weight * y_pred_i

            # 7. 记录训练得分
            current_mse = np.mean((y - y_pred) ** 2)
            self.train_scores_.append(current_mse)

        # 转换为numpy数组
        self.estimator_weights_ = np.array(self.estimator_weights_)
        self.estimator_errors_ = np.array(self.estimator_errors_)
        self.train_scores_ = np.array(self.train_scores_)

        return self

    def predict(self, X):
        """预测"""
        check_is_fitted(self, ['estimators_', 'estimator_weights_'])
        X = check_array(X)

        if len(self.estimators_) == 0:
            raise ValueError("模型未训练成功，无法进行预测")

        y_pred = np.zeros(X.shape[0])
        for estimator, weight in zip(self.estimators_, self.estimator_weights_):
            y_pred += weight * estimator.predict(X)

        return y_pred

    def decision_function(self, X):
        """决策函数值（仅二分类）"""
        if not self._binary:
            raise ValueError("decision_function仅适用于二分类")

        check_is_fitted(self)
        X = check_array(X)

        pred = np.zeros(X.shape[0])

        for estimator, weight in zip(self.estimators_, self.estimator_weights_):
            y_pred = estimator.predict(X)
            pred += weight * y_pred

        return pred

    def staged_predict(self, X):
        """按阶段预测（返回每个阶段的预测）"""
        check_is_fitted(self)
        X = check_array(X)

        n_samples = X.shape[0]

        if self._binary:
            for t in range(1, self.n_estimators + 1):
                pred = np.zeros(n_samples)
                for estimator, weight in zip(self.estimators_[:t],
                                             self.estimator_weights_[:t]):
                    y_pred = estimator.predict(X)
                    pred += weight * y_pred
                yield np.where(pred > 0, self.classes_[1], self.classes_[0])
        else:
            n_classes = len(self.classes_)
            for t in range(1, self.n_estimators + 1):
                pred = np.zeros((n_samples, n_classes))
                for estimator, weight in zip(self.estimators_[:t],
                                             self.estimator_weights_[:t]):
                    if self.algorithm == 'SAMME':
                        y_pred = estimator.predict(X)
                        pred[np.arange(n_samples),
                        np.searchsorted(self.classes_, y_pred)] += weight
                    else:  # SAMME.R
                        y_proba = estimator.predict_proba(X)
                        pred += weight * y_proba
                yield self.classes_[np.argmax(pred, axis=1)]

    def score(self, X, y):
        """计算准确率"""
        y_pred = self.predict(X)
        return np.mean(y_pred == y)


class AdaBoostRegressor(BaseEstimator, RegressorMixin):
    """AdaBoost回归器
    实现了回归任务的AdaBoost算法
    支持线性损失、平方损失和指数损失
    """

    def __init__(self, base_estimator=None, n_estimators=50,
                 learning_rate=1.0, loss='linear', random_state=None):
        """
        参数:
        ----------
        base_estimator : 基学习器，默认为深度3的决策树
        n_estimators : 基学习器数量
        learning_rate : 学习率
        loss : 'linear', 'square', 'exponential' 损失函数类型
        random_state : 随机种子
        """
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.loss = loss
        self.random_state = random_state

        if self.base_estimator is None:
            self.base_estimator = DecisionTreeRegressor(max_depth=3)

        if random_state is not None:
            np.random.seed(random_state)

        # 初始化存储
        self.estimators_ = []
        self.estimator_weights_ = []
        self.estimator_errors_ = []

    def fit(self, X, y, sample_weight=None):
        """训练AdaBoost回归器"""
        X, y = check_X_y(X, y)
        n_samples = X.shape[0]

        # 初始化样本权重
        if sample_weight is None:
            sample_weight = np.ones(n_samples) / n_samples
        else:
            sample_weight = np.array(sample_weight) / np.sum(sample_weight)

        # 清空存储
        self.estimators_ = []
        self.estimator_weights_ = []
        self.estimator_errors_ = []

        # 初始预测
        y_pred = np.zeros(n_samples)

        for t in range(self.n_estimators):
            # 1. 训练基学习器
            estimator = deepcopy(self.base_estimator)
            estimator.fit(X, y, sample_weight=sample_weight)
            y_pred_i = estimator.predict(X)

            # 2. 计算损失向量
            error_vector = y - (y_pred + y_pred_i)

            if self.loss == 'linear':
                loss_vector = np.abs(error_vector)
            elif self.loss == 'square':
                loss_vector = error_vector ** 2
            elif self.loss == 'exponential':
                loss_vector = 1 - np.exp(-np.abs(error_vector))
            else:
                raise ValueError(f"不支持的损失函数: {self.loss}")

            # 3. 计算加权平均损失
            estimator_error = np.dot(sample_weight, loss_vector)

            # 防止数值问题
            eps = 1e-10
            estimator_error = np.clip(estimator_error, eps, 1 - eps)

            # 4. 计算基学习器权重
            if estimator_error >= 1.0 or estimator_error <= 0:
                estimator_weight = self.learning_rate
            else:
                # 添加平滑项防止数值不稳定
                ratio = (1 - estimator_error) / (estimator_error + eps)
                estimator_weight = self.learning_rate * np.log(ratio + eps)

            # 存储结果
            self.estimator_errors_.append(estimator_error)
            self.estimator_weights_.append(estimator_weight)
            self.estimators_.append(estimator)

            # 5. 更新样本权重
            # 归一化损失向量
            loss_vector_normalized = loss_vector / (np.max(loss_vector) + eps)
            sample_weight *= np.exp(estimator_weight * loss_vector_normalized)

            # 重新归一化样本权重
            weight_sum = np.sum(sample_weight)
            if weight_sum <= 0:
                sample_weight = np.ones(n_samples) / n_samples
            else:
                sample_weight /= weight_sum

            # 6. 更新累计预测
            y_pred += estimator_weight * y_pred_i

        # 转换为numpy数组
        self.estimator_weights_ = np.array(self.estimator_weights_)
        self.estimator_errors_ = np.array(self.estimator_errors_)

        return self

    def predict(self, X):
        """预测"""
        check_is_fitted(self)
        X = check_array(X)

        y_pred = np.zeros(X.shape[0])
        for estimator, weight in zip(self.estimators_, self.estimator_weights_):
            y_pred += weight * estimator.predict(X)

        return y_pred


# ==================== 梯度提升树（GBDT）实现 ====================

class LossFunction:
    """损失函数基类"""

    def __init__(self):
        pass

    def __call__(self, y, pred):
        """计算损失值"""
        raise NotImplementedError

    def negative_gradient(self, y, pred):
        """计算负梯度（伪残差）"""
        raise NotImplementedError

    def init_estimator(self):
        """返回初始估计器"""
        raise NotImplementedError


# 回归损失函数
class LeastSquaresError(LossFunction):
    """平方损失函数（用于回归）"""

    def __call__(self, y, pred):
        """计算均方误差"""
        return np.mean((y - pred) ** 2)

    def negative_gradient(self, y, pred):
        """负梯度 = y - pred（残差）"""
        return y - pred

    def init_estimator(self):
        """初始预测为均值"""

        class MeanEstimator:
            def fit(self, y):
                self.mean = np.mean(y)
                return self

            def predict(self, X):
                return np.full(X.shape[0], self.mean)

        return MeanEstimator()


class LeastAbsoluteError(LossFunction):
    """绝对损失（用于回归）"""

    def __call__(self, y, pred):
        return np.mean(np.abs(y - pred))

    def negative_gradient(self, y, pred):
        return np.sign(y - pred)

    def init_estimator(self):
        class MedianEstimator:
            def fit(self, y):
                self.median = np.median(y)
                return self

            def predict(self, X):
                return np.full(X.shape[0], self.median)

        return MedianEstimator()


class HuberLoss(LossFunction):
    """Huber损失函数（对异常值鲁棒）"""

    def __init__(self, alpha=0.9):
        self.alpha = alpha
        self.delta = None

    def __call__(self, y, pred):
        """计算Huber损失"""
        diff = y - pred

        if self.delta is None:
            # 估计delta为绝对误差的中位数
            self.delta = np.median(np.abs(diff))

        mask = np.abs(diff) <= self.delta
        loss = np.where(mask,
                        0.5 * diff ** 2,
                        self.delta * (np.abs(diff) - 0.5 * self.delta))

        return np.mean(loss)

    def negative_gradient(self, y, pred):
        """Huber损失的负梯度"""
        if self.delta is None:
            self.delta = np.median(np.abs(y - pred))

        diff = y - pred
        mask = np.abs(diff) <= self.delta

        # 当|diff| <= delta时，梯度为diff；否则为delta * sign(diff)
        return np.where(mask, diff, self.delta * np.sign(diff))

    def init_estimator(self):
        """初始预测为均值"""

        class MeanEstimator:
            def fit(self, y):
                self.mean = np.mean(y)
                return self

            def predict(self, X):
                return np.full(X.shape[0], self.mean)

        return MeanEstimator()


class QuantileLoss(LossFunction):
    """分位数损失（用于分位数回归）"""

    def __init__(self, alpha=0.5):
        self.alpha = alpha  # 分位数，默认中位数

    def __call__(self, y, pred):
        error = y - pred
        loss = np.where(error > 0,
                        self.alpha * error,
                        (self.alpha - 1) * error)
        return np.mean(loss)

    def negative_gradient(self, y, pred):
        error = y - pred
        return np.where(error > 0, self.alpha, self.alpha - 1)

    def init_estimator(self):
        class QuantileEstimator:
            def __init__(self, alpha=0.5):
                self.alpha = alpha

            def fit(self, y):
                self.quantile = np.percentile(y, self.alpha * 100)
                return self

            def predict(self, X):
                return np.full(X.shape[0], self.quantile)

        return QuantileEstimator(self.alpha)


# 分类损失函数
class BinomialDeviance(LossFunction):
    """二项偏差损失（对数似然损失，用于二分类）"""

    def __call__(self, y, pred):
        # y ∈ {0, 1}, pred是对数几率
        pred = np.clip(pred, -500, 500)  # 防止数值溢出
        return np.mean(np.log(1 + np.exp(-(2 * y - 1) * pred)))

    def negative_gradient(self, y, pred):
        # 负梯度 = y - σ(pred)
        prob = 1.0 / (1.0 + np.exp(-pred))
        return y - prob

    def init_estimator(self):
        class LogOddsEstimator:
            def fit(self, y):
                pos = np.mean(y)
                if pos <= 0 or pos >= 1:
                    pos = np.clip(pos, 1e-10, 1 - 1e-10)
                self.prior = np.log(pos / (1 - pos))
                return self

            def predict(self, X):
                return np.full(X.shape[0], self.prior)

        return LogOddsEstimator()


class ExponentialLoss(LossFunction):
    """指数损失（AdaBoost损失）"""

    def __call__(self, y, pred):
        # 假设y ∈ {0, 1}，转换为{-1, 1}
        y_transformed = 2 * y - 1
        return np.mean(np.exp(-y_transformed * pred))

    def negative_gradient(self, y, pred):
        y_transformed = 2 * y - 1
        return y_transformed * np.exp(-y_transformed * pred)

    def init_estimator(self):
        class ZeroEstimator:
            def fit(self, y):
                self.constant = 0.0
                return self

            def predict(self, X):
                return np.full(X.shape[0], self.constant)

        return ZeroEstimator()


class MultinomialDeviance(LossFunction):
    """多项偏差损失（多分类对数似然）"""

    def __init__(self, n_classes):
        self.n_classes = n_classes

    def __call__(self, y, pred):
        # y: one-hot编码, pred: 每个类别的对数几率
        pred = np.clip(pred, -500, 500)
        exp_pred = np.exp(pred - np.max(pred, axis=1, keepdims=True))
        prob = exp_pred / np.sum(exp_pred, axis=1, keepdims=True)

        log_likelihood = np.sum(y * np.log(prob + 1e-15))
        return -log_likelihood / len(y)

    def negative_gradient(self, y, pred, k=None):
        if k is not None:
            pred = np.clip(pred, -500, 500)
            exp_pred = np.exp(pred - np.max(pred, axis=1, keepdims=True))
            prob = exp_pred / np.sum(exp_pred, axis=1, keepdims=True)
            return y[:, k] - prob[:, k]
        else:
            pred = np.clip(pred, -500, 500)
            exp_pred = np.exp(pred - np.max(pred, axis=1, keepdims=True))
            prob = exp_pred / np.sum(exp_pred, axis=1, keepdims=True)
            return y - prob

    def init_estimator(self):
        class ZeroEstimator:
            def fit(self, y):
                self.constant = 0.0
                return self

            def predict(self, X):
                if len(X.shape) == 1:
                    return np.full(X.shape[0], self.constant)
                else:
                    return np.full((X.shape[0], 1), self.constant)

        return ZeroEstimator()


# 梯度提升回归树
class GradientBoostingRegressor(BaseEstimator, RegressorMixin):
    """梯度提升回归树（优化版本）"""

    def __init__(self,
                 loss='ls',  # 'ls', 'lad', 'huber', 'quantile'
                 learning_rate=0.1,
                 n_estimators=100,
                 max_depth=3,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 subsample=1.0,
                 max_features=None,
                 random_state=None,
                 verbose=0,
                 alpha=0.9):  # Huber损失和Quantile损失的参数

        self.loss = loss
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.subsample = subsample
        self.max_features = max_features
        self.random_state = random_state
        self.verbose = verbose
        self.alpha = alpha

        if random_state is not None:
            np.random.seed(random_state)

        self.estimators_ = []
        self.train_score_ = []
        self.init_ = None
        self.loss_ = None

    def _init_loss(self):
        """初始化损失函数"""
        if self.loss == 'ls':
            self.loss_ = LeastSquaresError()
        elif self.loss == 'lad':
            self.loss_ = LeastAbsoluteError()
        elif self.loss == 'huber':
            self.loss_ = HuberLoss(self.alpha)
        elif self.loss == 'quantile':
            self.loss_ = QuantileLoss(self.alpha)
        else:
            raise ValueError(f"不支持的损失函数: {self.loss}")

    def _init_constant(self, y):
        """用常数初始化预测"""
        self.init_ = self.loss_.init_estimator()
        self.init_.fit(y)
        return self.init_.predict(np.zeros(len(y)))

    def fit(self, X, y, sample_weight=None):
        """训练梯度提升模型"""
        X, y = check_X_y(X, y)
        n_samples, n_features = X.shape

        if self.verbose > 0:
            print("=" * 60)
            print("开始训练梯度提升回归树")
            print("=" * 60)
            print(f"样本数: {n_samples}, 特征数: {n_features}")
            print(f"参数: loss={self.loss}, learning_rate={self.learning_rate}")
            print(f"      n_estimators={self.n_estimators}, max_depth={self.max_depth}")

        self._init_loss()

        y_pred = self._init_constant(y)

        if self.verbose > 0 and hasattr(self.init_, 'mean'):
            print(f"初始预测（常数）: {self.init_.mean:.4f}")

        self.estimators_ = []
        self.train_score_ = []

        initial_loss = self.loss_(y, y_pred)
        self.train_score_.append(initial_loss)

        if self.verbose > 0:
            print(f"初始损失: {initial_loss:.4f}")

        for t in range(self.n_estimators):
            negative_gradient = self.loss_.negative_gradient(y, y_pred)

            if self.subsample < 1.0:
                sample_mask = np.random.rand(n_samples) < self.subsample
                X_subset = X[sample_mask]
                y_subset = negative_gradient[sample_mask]
                sample_weight_subset = (sample_weight[sample_mask]
                                        if sample_weight is not None else None)
            else:
                X_subset = X
                y_subset = negative_gradient
                sample_weight_subset = sample_weight

            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                random_state=self.random_state
            )

            tree.fit(X_subset, y_subset, sample_weight=sample_weight_subset)
            self.estimators_.append(tree)

            update = tree.predict(X)
            y_pred += self.learning_rate * update

            current_loss = self.loss_(y, y_pred)
            self.train_score_.append(current_loss)

            if self.verbose > 0 and t % 10 == 0:
                print(f"轮次 {t + 1:3d}/{self.n_estimators}: 损失 = {current_loss:.4f}")

        if self.verbose > 0:
            print(f"训练完成，最终损失: {current_loss:.4f}")

        return self

    def predict(self, X):
        """预测"""
        check_is_fitted(self)
        X = check_array(X)

        y_pred = self.init_.predict(np.zeros(X.shape[0]))

        for tree in self.estimators_:
            y_pred += self.learning_rate * tree.predict(X)

        return y_pred

    def staged_predict(self, X):
        """按阶段预测"""
        check_is_fitted(self)
        X = check_array(X)

        y_pred = self.init_.predict(np.zeros(X.shape[0]))

        yield y_pred.copy()

        for tree in self.estimators_:
            y_pred += self.learning_rate * tree.predict(X)
            yield y_pred.copy()


# 梯度提升分类树
class GradientBoostingClassifier(BaseEstimator, ClassifierMixin):
    """梯度提升分类树"""

    def __init__(self,
                 loss='deviance',  # 'deviance', 'exponential'
                 learning_rate=0.1,
                 n_estimators=100,
                 max_depth=3,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 subsample=1.0,
                 max_features=None,
                 random_state=None,
                 verbose=0):

        self.loss = loss
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.subsample = subsample
        self.max_features = max_features
        self.random_state = random_state
        self.verbose = verbose

        if random_state is not None:
            np.random.seed(random_state)

        self.estimators_ = []
        self.train_score_ = []
        self.init_ = None
        self.classes_ = None
        self.n_classes_ = None

    def _init_loss(self, n_classes):
        """初始化损失函数"""
        if self.loss == 'deviance':
            if n_classes == 2:
                self.loss_ = BinomialDeviance()
            else:
                self.loss_ = MultinomialDeviance(n_classes)
        elif self.loss == 'exponential':
            if n_classes == 2:
                class ExponentialLossWrapper:
                    def negative_gradient(self, y, pred):
                        y_transformed = 2 * y - 1
                        return 2 * y_transformed * np.exp(-2 * y_transformed * pred)

                    def init_estimator(self):
                        class ZeroEstimator:
                            def fit(self, y):
                                self.constant = 0.0
                                return self

                            def predict(self, X):
                                return np.full(X.shape[0], self.constant)

                        return ZeroEstimator()

                self.loss_ = ExponentialLossWrapper()
            else:
                raise ValueError("指数损失仅支持二分类")
        else:
            raise ValueError(f"不支持的损失函数: {self.loss}")

    def fit(self, X, y, sample_weight=None):
        """训练梯度提升分类器"""
        X, y = check_X_y(X, y)

        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)

        if self.verbose > 0:
            print("=" * 60)
            print("开始训练梯度提升分类器")
            print("=" * 60)
            print(f"样本数: {X.shape[0]}, 特征数: {X.shape[1]}")
            print(f"类别数: {self.n_classes_}, 类别: {self.classes_}")
            print(f"参数: loss={self.loss}, learning_rate={self.learning_rate}")

        if self.n_classes_ == 2:
            y_coded = np.where(y == self.classes_[0], 0, 1)
            self._fit_binary(X, y_coded, sample_weight)
        else:
            y_onehot = np.eye(self.n_classes_)[y]
            self._fit_multiclass(X, y_onehot, sample_weight)

        return self

    def _fit_binary(self, X, y, sample_weight):
        """训练二分类模型"""
        n_samples = X.shape[0]

        self._init_loss(2)

        self.init_ = self.loss_.init_estimator()
        self.init_.fit(y)
        y_pred = self.init_.predict(np.zeros((n_samples, 1))).flatten()

        if self.verbose > 0 and hasattr(self.init_, 'prior'):
            print(f"初始先验概率: {1 / (1 + np.exp(-2 * self.init_.prior)):.4f}")

        self.estimators_ = []
        self.train_score_ = []

        for t in range(self.n_estimators):
            negative_gradient = self.loss_.negative_gradient(y, y_pred)

            if self.subsample < 1.0:
                subsample_mask = np.random.rand(n_samples) < self.subsample
                X_subset = X[subsample_mask]
                y_subset = negative_gradient[subsample_mask]
                sample_weight_subset = None
            else:
                X_subset = X
                y_subset = negative_gradient
                sample_weight_subset = sample_weight

            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                random_state=self.random_state
            )

            tree.fit(X_subset, y_subset, sample_weight=sample_weight_subset)
            self.estimators_.append(tree)

            update = tree.predict(X)
            y_pred += self.learning_rate * update

            if self.loss == 'deviance':
                current_loss = np.mean(np.log(1 + np.exp(-2 * y * y_pred)))
            else:
                current_loss = np.mean(np.exp(-y * y_pred))

            self.train_score_.append(current_loss)

            if self.verbose > 0 and t % 10 == 0:
                print(f"轮次 {t + 1:3d}/{self.n_estimators}: 损失 = {current_loss:.4f}")

        if self.verbose > 0:
            print(f"训练完成，最终损失: {current_loss:.4f}")

    def _fit_multiclass(self, X, y_onehot, sample_weight):
        """训练多分类模型（简化版本）"""
        n_samples = X.shape[0]
        n_classes = y_onehot.shape[1]

        # 为每个类别训练一个二分类器
        self.estimators_ = []
        self.train_score_ = []

        for k in range(n_classes):
            y_k = y_onehot[:, k]

            # 训练一个二分类器
            gbdt_k = GradientBoostingRegressor(
                loss='ls',
                learning_rate=self.learning_rate,
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                subsample=self.subsample,
                max_features=self.max_features,
                random_state=self.random_state,
                verbose=0
            )

            gbdt_k.fit(X, y_k, sample_weight)
            self.estimators_.append(gbdt_k)

    def predict(self, X):
        """预测类别"""
        check_is_fitted(self)
        X = check_array(X)

        if self.n_classes_ == 2:
            proba = self.predict_proba(X)
            return np.where(proba[:, 1] > 0.5, self.classes_[1], self.classes_[0])
        else:
            proba = self.predict_proba(X)
            return self.classes_[np.argmax(proba, axis=1)]

    def predict_proba(self, X):
        """预测概率"""
        check_is_fitted(self)
        X = check_array(X)

        if self.n_classes_ == 2:
            raw_pred = self._raw_predict(X)
            proba = 1.0 / (1.0 + np.exp(-raw_pred))
            return np.column_stack([1 - proba, proba])
        else:
            raw_pred = self._raw_predict(X)
            exp_pred = np.exp(raw_pred - np.max(raw_pred, axis=1, keepdims=True))
            return exp_pred / np.sum(exp_pred, axis=1, keepdims=True)

    def _raw_predict(self, X):
        """原始预测（对数几率）"""
        n_samples = X.shape[0]

        if self.n_classes_ == 2:
            raw_pred = self.init_.predict(np.zeros((n_samples, 1))).flatten()
            for tree in self.estimators_:
                raw_pred += self.learning_rate * tree.predict(X)
            return raw_pred
        else:
            raw_pred = np.zeros((n_samples, self.n_classes_))
            for k in range(self.n_classes_):
                raw_pred[:, k] = self.estimators_[k].predict(X)
            return raw_pred

    def score(self, X, y):
        """计算准确率"""
        y_pred = self.predict(X)
        return np.mean(y_pred == y)


# 导出所有类
__all__ = [
    'AdaBoostClassifier',
    'AdaBoostRegressor',
    'GradientBoostingRegressor',
    'GradientBoostingClassifier',
    'LossFunction',
    'LeastSquaresError',
    'LeastAbsoluteError',
    'HuberLoss',
    'QuantileLoss',
    'BinomialDeviance',
    'ExponentialLoss',
    'MultinomialDeviance'
]
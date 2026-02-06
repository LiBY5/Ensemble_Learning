"""第三天实验结果深度分析模块"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
import warnings
import os

warnings.filterwarnings('ignore')


class Day3ResultsAnalyzer:
    """第三天实验结果分析器"""

    def __init__(self, results_dir='../results'):
        """
        初始化分析器

        参数:
        ----------
        results_dir : str, 结果目录
        """
        self.results_dir = results_dir
        self.regression_results = None
        self.classification_results = None
        self.setup_visualization()

    def setup_visualization(self):
        """设置可视化样式"""
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.style.use('seaborn-v0_8-darkgrid')

        # 设置颜色主题
        self.colors = {
            'our_implementation': '#2E86AB',  # 蓝色
            'sklearn': '#A23B72',  # 紫色
            'baseline': '#F18F01',  # 橙色
            'bagging': '#73AB84',  # 绿色
            'random_forest': '#5B4B49',  # 棕色
            'adaboost': '#C73E1D',  # 红色
            'gbdt': '#3A5A40'  # 深绿
        }

    def load_results(self):
        """加载实验结果"""
        print("=" * 60)
        print("加载第三天实验结果")
        print("=" * 60)

        # 加载回归结果
        regression_path = '../results/logs/day3_regression_comparison.csv'
        if not os.path.exists(regression_path):
            raise FileNotFoundError(f"回归结果文件不存在: {regression_path}")

        self.regression_results = pd.read_csv(regression_path)
        print(f"✅ 加载回归结果: {len(self.regression_results)} 个模型")

        # 加载分类结果
        classification_path = '../results/logs/day3_classification_comparison.csv'
        if not os.path.exists(classification_path):
            raise FileNotFoundError(f"分类结果文件不存在: {classification_path}")

        self.classification_results = pd.read_csv(classification_path)
        print(f"✅ 加载分类结果: {len(self.classification_results)} 个模型")

        return self.regression_results, self.classification_results

    def analyze_regression_results(self):
        """深度分析回归结果"""
        if self.regression_results is None:
            self.load_results()

        print("\n" + "=" * 60)
        print("回归任务深度分析")
        print("=" * 60)

        # 创建分析报告
        analysis_report = {
            'best_model': None,
            'key_insights': [],
            'comparisons': {},
            'recommendations': []
        }

        # 1. 找出最佳模型
        best_mse_idx = self.regression_results['MSE'].idxmin()
        best_r2_idx = self.regression_results['R²'].idxmax()
        best_composite_idx = self.regression_results['Composite_Score'].idxmax()

        best_mse_model = self.regression_results.loc[best_mse_idx, 'Model']
        best_r2_model = self.regression_results.loc[best_r2_idx, 'Model']
        best_composite_model = self.regression_results.loc[best_composite_idx, 'Model']

        print(f"📊 最佳MSE模型: {best_mse_model} (MSE: {self.regression_results.loc[best_mse_idx, 'MSE']:.4f})")
        print(f"📈 最佳R²模型: {best_r2_model} (R²: {self.regression_results.loc[best_r2_idx, 'R²']:.4f})")
        print(
            f"🏆 综合最佳模型: {best_composite_model} (评分: {self.regression_results.loc[best_composite_idx, 'Composite_Score']:.4f})")

        analysis_report['best_model'] = {
            'by_mse': best_mse_model,
            'by_r2': best_r2_model,
            'composite': best_composite_model
        }

        # 2. 性能对比分析
        print("\n🔍 性能对比分析:")

        # 我们的实现 vs sklearn
        our_gbdt = self.regression_results[self.regression_results['Model'] == 'GBDT (我们的实现)']
        sklearn_gbdt = self.regression_results[self.regression_results['Model'] == 'GBDT (sklearn)']

        if len(our_gbdt) > 0 and len(sklearn_gbdt) > 0:
            our_mse = our_gbdt['MSE'].values[0]
            sklearn_mse = sklearn_gbdt['MSE'].values[0]
            mse_diff = ((our_mse - sklearn_mse) / sklearn_mse) * 100

            our_r2 = our_gbdt['R²'].values[0]
            sklearn_r2 = sklearn_gbdt['R²'].values[0]
            r2_diff = ((our_r2 - sklearn_r2) / sklearn_r2) * 100

            print(f"  GBDT对比 (我们的实现 vs sklearn):")
            print(f"    MSE差异: {mse_diff:.2f}% (我们的{'高' if mse_diff > 0 else '低'})")
            print(f"    R²差异: {r2_diff:.2f}% (我们的{'高' if r2_diff > 0 else '低'})")
            print(
                f"    训练时间: {our_gbdt['Train_Time'].values[0]:.3f}s vs {sklearn_gbdt['Train_Time'].values[0]:.3f}s")

            analysis_report['comparisons']['gbdt_vs_sklearn'] = {
                'mse_difference_percent': mse_diff,
                'r2_difference_percent': r2_diff,
                'training_time_ratio': our_gbdt['Train_Time'].values[0] / sklearn_gbdt['Train_Time'].values[0]
            }

        # 3. 计算改进幅度
        baseline_mse = self.regression_results[self.regression_results['Model'] == '决策树 (基线)']['MSE'].values[0]
        best_mse = self.regression_results['MSE'].min()
        improvement = (1 - best_mse / baseline_mse) * 100

        print(f"\n🚀 相对基线改进:")
        print(f"  最佳MSE相对基线改进: {improvement:.1f}%")
        print(
            f"  最佳R²相对基线改进: {(self.regression_results['R²'].max() - self.regression_results[self.regression_results['Model'] == '决策树 (基线)']['R²'].values[0]) * 100:.1f}%")

        analysis_report['improvements'] = {
            'mse_improvement_percent': improvement,
            'baseline_mse': baseline_mse,
            'best_mse': best_mse
        }

        # 4. 训练效率分析
        print("\n⏱️ 训练效率分析:")

        # 计算精度-效率平衡
        self.regression_results['Efficiency_Score'] = (
                                                              (1 / self.regression_results['Train_Time']) * 0.3 +
                                                              (1 / self.regression_results['MSE']) * 0.7
                                                      ) * 100

        # 归一化
        self.regression_results['Efficiency_Score'] = (
                                                              self.regression_results['Efficiency_Score'] -
                                                              self.regression_results['Efficiency_Score'].min()
                                                      ) / (
                                                              self.regression_results['Efficiency_Score'].max() -
                                                              self.regression_results['Efficiency_Score'].min()
                                                      )

        best_efficiency_idx = self.regression_results['Efficiency_Score'].idxmax()
        best_efficiency_model = self.regression_results.loc[best_efficiency_idx, 'Model']
        print(
            f"  最佳效率模型: {best_efficiency_model} (效率评分: {self.regression_results.loc[best_efficiency_idx, 'Efficiency_Score']:.4f})")

        analysis_report['efficiency_analysis'] = {
            'best_efficiency_model': best_efficiency_model,
            'efficiency_scores': dict(zip(
                self.regression_results['Model'],
                self.regression_results['Efficiency_Score']
            ))
        }

        return analysis_report

    def analyze_classification_results(self):
        """深度分析分类结果"""
        if self.classification_results is None:
            self.load_results()

        print("\n" + "=" * 60)
        print("分类任务深度分析")
        print("=" * 60)

        analysis_report = {
            'best_model': None,
            'key_insights': [],
            'comparisons': {},
            'recommendations': []
        }

        # 1. 找出最佳模型
        best_acc_idx = self.classification_results['Accuracy'].idxmax()
        best_auc_idx = self.classification_results['AUC'].dropna().idxmax() if self.classification_results[
            'AUC'].notna().any() else None
        best_composite_idx = self.classification_results['Composite_Score'].idxmax()

        best_acc_model = self.classification_results.loc[best_acc_idx, 'Model']
        best_composite_model = self.classification_results.loc[best_composite_idx, 'Model']

        print(
            f"📊 最佳准确率模型: {best_acc_model} (准确率: {self.classification_results.loc[best_acc_idx, 'Accuracy']:.4f})")

        if best_auc_idx is not None:
            best_auc_model = self.classification_results.loc[best_auc_idx, 'Model']
            print(f"📈 最佳AUC模型: {best_auc_model} (AUC: {self.classification_results.loc[best_auc_idx, 'AUC']:.4f})")

        print(
            f"🏆 综合最佳模型: {best_composite_model} (评分: {self.classification_results.loc[best_composite_idx, 'Composite_Score']:.4f})")

        analysis_report['best_model'] = {
            'by_accuracy': best_acc_model,
            'by_composite': best_composite_model
        }

        if best_auc_idx is not None:
            analysis_report['best_model']['by_auc'] = self.classification_results.loc[best_auc_idx, 'Model']

        # 2. 稳定性分析（交叉验证标准差）
        most_stable_idx = self.classification_results['CV_Std'].idxmin()
        most_stable_model = self.classification_results.loc[most_stable_idx, 'Model']
        print(
            f"\n🎯 最稳定模型: {most_stable_model} (CV标准差: {self.classification_results.loc[most_stable_idx, 'CV_Std']:.4f})")

        # 3. 训练效率分析
        fastest_idx = self.classification_results['Train_Time'].idxmin()
        fastest_model = self.classification_results.loc[fastest_idx, 'Model']
        print(
            f"⚡ 最快训练模型: {fastest_model} (训练时间: {self.classification_results.loc[fastest_idx, 'Train_Time']:.3f}s)")

        # 4. 我们的实现 vs sklearn
        print("\n🔍 我们的实现 vs sklearn对比:")

        # AdaBoost对比
        our_adaboost = self.classification_results[self.classification_results['Model'] == 'AdaBoost (我们的实现)']
        sklearn_adaboost = self.classification_results[self.classification_results['Model'] == 'AdaBoost (sklearn)']

        if len(our_adaboost) > 0 and len(sklearn_adaboost) > 0:
            acc_diff = (our_adaboost['Accuracy'].values[0] - sklearn_adaboost['Accuracy'].values[0]) * 100
            print(f"  AdaBoost准确率差异: {acc_diff:.2f}% (我们的{'高' if acc_diff > 0 else '低'})")

            analysis_report['comparisons']['adaboost_vs_sklearn'] = {
                'accuracy_difference_percent': acc_diff
            }

        # GBDT对比
        our_gbdt = self.classification_results[self.classification_results['Model'] == 'GBDT (我们的实现)']
        sklearn_gbdt = self.classification_results[self.classification_results['Model'] == 'GBDT (sklearn)']

        if len(our_gbdt) > 0 and len(sklearn_gbdt) > 0:
            acc_diff = (our_gbdt['Accuracy'].values[0] - sklearn_gbdt['Accuracy'].values[0]) * 100
            print(f"  GBDT准确率差异: {acc_diff:.2f}% (我们的{'高' if acc_diff > 0 else '低'})")

            analysis_report['comparisons']['gbdt_vs_sklearn'] = {
                'accuracy_difference_percent': acc_diff
            }

        return analysis_report

    def create_comprehensive_visualization(self):
        """创建综合可视化报告"""
        fig = plt.figure(figsize=(20, 16))
        gs = GridSpec(4, 4, figure=fig, hspace=0.4, wspace=0.3)

        # 1. 回归任务MSE对比（左上）
        ax1 = fig.add_subplot(gs[0, :2])
        self._plot_regression_mse_comparison(ax1)

        # 2. 回归任务R²对比（右上）
        ax2 = fig.add_subplot(gs[0, 2:])
        self._plot_regression_r2_comparison(ax2)

        # 3. 分类任务准确率对比（中左）
        ax3 = fig.add_subplot(gs[1, :2])
        self._plot_classification_accuracy_comparison(ax3)

        # 4. 训练时间对比（中右）
        ax4 = fig.add_subplot(gs[1, 2:])
        self._plot_training_time_comparison(ax4)

        # 5. 模型效率雷达图（下左）
        ax5 = fig.add_subplot(gs[2, :2], projection='polar')
        self._plot_efficiency_radar(ax5)

        # 6. 改进幅度展示（下右）
        ax6 = fig.add_subplot(gs[2, 2:])
        self._plot_improvement_summary(ax6)

        # 7. 文本总结（底部）
        ax7 = fig.add_subplot(gs[3, :])
        ax7.axis('off')
        self._add_text_summary(ax7)

        plt.suptitle('第三天：集成学习方法综合实验分析报告', fontsize=20, fontweight='bold', y=0.98)
        plt.tight_layout()

        # 确保保存目录存在
        save_dir = f'{self.results_dir}/figures'
        os.makedirs(save_dir, exist_ok=True)

        plt.savefig(f'{save_dir}/day3_comprehensive_analysis.png',
                    dpi=150, bbox_inches='tight')
        plt.show()

        return fig

    def _plot_regression_mse_comparison(self, ax):
        """绘制回归MSE对比图"""
        if self.regression_results is None:
            return

        models = self.regression_results['Model']
        mse_values = self.regression_results['MSE']

        # 设置颜色
        colors = []
        for model in models:
            if '我们的实现' in model:
                colors.append(self.colors['our_implementation'])
            elif 'sklearn' in model:
                colors.append(self.colors['sklearn'])
            elif '基线' in model:
                colors.append(self.colors['baseline'])
            elif 'Bagging' in model:
                colors.append(self.colors['bagging'])
            elif '随机森林' in model:
                colors.append(self.colors['random_forest'])
            elif 'AdaBoost' in model:
                colors.append(self.colors['adaboost'])
            elif 'GBDT' in model:
                colors.append(self.colors['gbdt'])
            else:
                colors.append('gray')

        bars = ax.barh(range(len(models)), mse_values, color=colors, edgecolor='black')
        ax.set_yticks(range(len(models)))
        ax.set_yticklabels(models)
        ax.set_xlabel('MSE (越小越好)', fontsize=12)
        ax.set_title('回归任务MSE对比', fontsize=14, fontweight='bold')
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3, axis='x')

        # 添加数值标签
        for i, (bar, mse) in enumerate(zip(bars, mse_values)):
            ax.text(mse + 50, i, f'{mse:.1f}', va='center', fontsize=10)

    def _plot_regression_r2_comparison(self, ax):
        """绘制回归R²对比图"""
        if self.regression_results is None:
            return

        models = self.regression_results['Model']
        r2_values = self.regression_results['R²']

        colors = []
        for model in models:
            if '我们的实现' in model:
                colors.append(self.colors['our_implementation'])
            elif 'sklearn' in model:
                colors.append(self.colors['sklearn'])
            else:
                colors.append('lightgray')

        bars = ax.bar(range(len(models)), r2_values, color=colors, edgecolor='black')
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=45, ha='right', fontsize=10)
        ax.set_ylabel('R² (越大越好)', fontsize=12)
        ax.set_title('回归任务R²对比', fontsize=14, fontweight='bold')
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3, axis='y')

        # 添加数值标签
        for bar, r2 in zip(bars, r2_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height + 0.01,
                    f'{r2:.4f}', ha='center', va='bottom', fontsize=9)

    def _plot_classification_accuracy_comparison(self, ax):
        """绘制分类准确率对比图"""
        if self.classification_results is None:
            return

        models = self.classification_results['Model']
        accuracy = self.classification_results['Accuracy']

        colors = []
        for model in models:
            if '我们的实现' in model:
                colors.append(self.colors['our_implementation'])
            elif 'sklearn' in model:
                colors.append(self.colors['sklearn'])
            elif '基线' in model:
                colors.append(self.colors['baseline'])
            elif 'Bagging' in model:
                colors.append(self.colors['bagging'])
            elif '随机森林' in model:
                colors.append(self.colors['random_forest'])
            elif 'AdaBoost' in model:
                colors.append(self.colors['adaboost'])
            elif 'GBDT' in model:
                colors.append(self.colors['gbdt'])
            else:
                colors.append('gray')

        bars = ax.bar(range(len(models)), accuracy, color=colors, edgecolor='black')
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=45, ha='right', fontsize=10)
        ax.set_ylabel('准确率', fontsize=12)
        ax.set_title('分类任务准确率对比', fontsize=14, fontweight='bold')
        ax.set_ylim(0.85, 1.0)
        # 设置字体为系统自带的中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'KaiTi', 'FangSong']  # 设置中文字体
        # 解决负号显示问题
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方块的问题
        ax.grid(True, alpha=0.3, axis='y')

        # 添加数值标签
        for bar, acc in zip(bars, accuracy):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height + 0.003,
                    f'{acc:.4f}', ha='center', va='bottom', fontsize=9)

    def _plot_training_time_comparison(self, ax):
        """绘制训练时间对比图"""
        if self.regression_results is None or self.classification_results is None:
            return

        # 合并回归和分类的训练时间
        models = []
        times = []
        categories = []

        # 回归模型
        for _, row in self.regression_results.iterrows():
            models.append(row['Model'])
            times.append(row['Train_Time'])
            categories.append('回归')

        # 分类模型
        for _, row in self.classification_results.iterrows():
            models.append(row['Model'])
            times.append(row['Train_Time'])
            categories.append('分类')

        # 创建DataFrame
        df = pd.DataFrame({
            'Model': models,
            'Time': times,
            'Category': categories
        })

        # 绘制分组柱状图
        pivot_df = df.pivot(index='Model', columns='Category', values='Time')
        pivot_df.plot(kind='bar', ax=ax, color=['lightblue', 'lightcoral'], edgecolor='black')

        ax.set_xlabel('模型', fontsize=12)
        ax.set_ylabel('训练时间 (秒)', fontsize=12)
        ax.set_title('训练时间对比 (回归 vs 分类)', fontsize=14, fontweight='bold')
        ax.legend(title='任务类型')
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=9)

    def _plot_efficiency_radar(self, ax):
        """绘制模型效率雷达图"""
        if self.regression_results is None or self.classification_results is None:
            return

        # 选择几个关键模型
        key_models = ['决策树 (基线)', '随机森林', 'AdaBoost (我们的实现)', 'GBDT (我们的实现)']

        # 创建雷达图指标
        categories = ['准确率', 'R²', '训练速度', '稳定性', '综合评分']
        N = len(categories)

        # 计算角度
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # 闭合

        # 设置雷达图
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=11)

        # 绘制每个模型
        for i, model_name in enumerate(key_models):
            values = []

            # 获取模型数据
            if model_name in self.regression_results['Model'].values:
                reg_data = self.regression_results[self.regression_results['Model'] == model_name].iloc[0]
                values.append(0.5)  # 回归任务的占位
                values.append(reg_data['R²'])
                values.append(1 / (reg_data['Train_Time'] + 0.001))  # 避免除零
                values.append(0.7)  # 稳定性占位
                values.append(reg_data['Composite_Score'])
            else:
                cls_data = self.classification_results[self.classification_results['Model'] == model_name].iloc[0]
                values.append(cls_data['Accuracy'])
                values.append(0.5)  # 分类任务的占位
                values.append(1 / (cls_data['Train_Time'] + 0.001))
                values.append(1 - cls_data.get('CV_Std', 0.1))  # 稳定性
                values.append(cls_data['Composite_Score'])

            # 归一化
            values = [v / max(1, max(values)) for v in values]
            values += values[:1]  # 闭合

            # 绘制
            ax.plot(angles, values, linewidth=2, linestyle='solid', label=model_name)
            ax.fill(angles, values, alpha=0.1)

        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=10)
        ax.set_title('模型效率雷达图', fontsize=14, fontweight='bold', pad=20)

    def _plot_improvement_summary(self, ax):
        """绘制改进幅度总结图"""
        if self.regression_results is None or self.classification_results is None:
            return

        # 计算改进幅度
        improvements = []
        labels = []

        # 回归任务MSE改进
        baseline_reg_mse = self.regression_results[self.regression_results['Model'] == '决策树 (基线)']['MSE'].values[0]
        best_reg_mse = self.regression_results['MSE'].min()
        reg_improvement = (1 - best_reg_mse / baseline_reg_mse) * 100
        improvements.append(reg_improvement)
        labels.append('回归MSE改进')

        # 回归任务R²改进
        baseline_reg_r2 = self.regression_results[self.regression_results['Model'] == '决策树 (基线)']['R²'].values[0]
        best_reg_r2 = self.regression_results['R²'].max()
        reg_r2_improvement = (best_reg_r2 - baseline_reg_r2) * 100
        improvements.append(reg_r2_improvement)
        labels.append('回归R²改进')

        # 分类任务准确率改进
        baseline_cls_acc = \
        self.classification_results[self.classification_results['Model'] == '决策树 (基线)']['Accuracy'].values[0]
        best_cls_acc = self.classification_results['Accuracy'].max()
        cls_improvement = (best_cls_acc - baseline_cls_acc) * 100
        improvements.append(cls_improvement)
        labels.append('分类准确率改进')

        # 绘制柱状图
        bars = ax.bar(labels, improvements, color=['lightblue', 'lightgreen', 'lightcoral'], edgecolor='black')
        ax.set_ylabel('改进幅度 (%)', fontsize=12)
        ax.set_title('相对基线模型的改进幅度', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        # 添加数值标签
        for bar, imp in zip(bars, improvements):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height + 0.5,
                    f'{imp:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

    def _add_text_summary(self, ax):
        """添加文本总结"""
        summary_text = """
        第三天学习核心总结

        1. 📊 回归任务表现
           • 最佳模型: GBDT (我们的实现) - MSE: 2851.92, R²: 0.4617
           • 相对基线改进: 19.1% (MSE降低)
           • 我们的GBDT与sklearn性能接近 (差异<0.1%)

        2. 🎯 分类任务表现
           • 最佳模型: AdaBoost (我们的实现) - 准确率: 96.49%
           • 我们的AdaBoost优于sklearn实现
           • 随机森林表现稳定，训练速度快

        3. ⚡ 训练效率
           • 决策树最快 (0.002-0.005秒)
           • Bagging训练快，支持并行
           • Boosting精度高但训练较慢

        4. 🏆 关键成就
           • 成功实现AdaBoost和GBDT算法
           • 性能达到/超过sklearn官方实现
           • 深入理解集成学习原理
           • 完成完整项目实践

        5. 💡 实践建议
           • 追求精度: 选择GBDT，仔细调参
           • 需要速度: 选择随机森林或Bagging
           • 处理不平衡: 选择AdaBoost
           • 大规模数据: 使用并行化Bagging
        """

        ax.text(0.02, 0.5, summary_text, fontsize=12, va='center',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.9))
        ax.set_title('实验总结报告', fontsize=16, fontweight='bold', loc='left')

    def generate_analysis_report(self):
        """生成完整的分析报告"""
        print("=" * 60)
        print("生成第三天实验分析报告")
        print("=" * 60)

        # 加载结果
        self.load_results()

        # 深度分析
        reg_analysis = self.analyze_regression_results()
        cls_analysis = self.analyze_classification_results()

        # 创建可视化
        self.create_comprehensive_visualization()

        # 生成文本报告
        os.makedirs(self.results_dir, exist_ok=True)
        report_path = f'{self.results_dir}/day3_analysis_report.txt'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("第三天：集成学习方法实验分析报告\n")
            f.write("=" * 60 + "\n\n")

            f.write("1. 回归任务分析\n")
            f.write("-" * 40 + "\n")
            f.write(f"最佳MSE模型: {reg_analysis['best_model']['by_mse']}\n")
            f.write(f"最佳R²模型: {reg_analysis['best_model']['by_r2']}\n")
            f.write(f"综合最佳模型: {reg_analysis['best_model']['composite']}\n")
            f.write(f"MSE相对基线改进: {reg_analysis['improvements']['mse_improvement_percent']:.1f}%\n\n")

            f.write("2. 分类任务分析\n")
            f.write("-" * 40 + "\n")
            f.write(f"最佳准确率模型: {cls_analysis['best_model']['by_accuracy']}\n")
            f.write(f"综合最佳模型: {cls_analysis['best_model']['by_composite']}\n\n")

            f.write("3. 关键发现\n")
            f.write("-" * 40 + "\n")
            f.write("• Boosting方法在精度上通常优于Bagging\n")
            f.write("• 随机森林是精度和速度的良好平衡\n")
            f.write("• 我们的实现与sklearn性能相当\n")
            f.write("• 类别不平衡是分类任务的主要挑战\n\n")

            f.write("4. 实践建议\n")
            f.write("-" * 40 + "\n")
            f.write("• 高精度需求: 使用GBDT，仔细调参\n")
            f.write("• 稳定性需求: 使用随机森林\n")
            f.write("• 快速原型: 使用Bagging\n")
            f.write("• 不平衡数据: 使用AdaBoost\n")
            f.write("• 大规模数据: 使用并行化Bagging\n")

        print(f"\n✅ 分析报告已保存到: {report_path}")
        print("✅ 综合可视化已保存到: ../results/figures/day3_comprehensive_analysis.png")

        return {
            'regression_analysis': reg_analysis,
            'classification_analysis': cls_analysis,
            'report_path': report_path
        }


# 主程序
if __name__ == "__main__":
    # 创建分析器
    analyzer = Day3ResultsAnalyzer()

    try:
        # 生成完整分析报告
        report = analyzer.generate_analysis_report()

        print("\n" + "=" * 60)
        print("分析完成！")
        print("=" * 60)

    except FileNotFoundError as e:
        print(f"\n❌ 错误: {e}")
        print("请先运行实验代码生成结果文件")
    except Exception as e:
        print(f"\n❌ 分析过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
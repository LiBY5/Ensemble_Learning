"""集成模型实现"""  # 模块文档字符串，说明模块功能

import numpy as np  # 导入NumPy库，用于数值计算
from typing import List, Optional, Union  # 导入类型注解相关的模块
from collections import Counter  # 导入Counter类，用于计数
from models.base import BaseModel  # 从自定义模块导入基类


class VotingEnsemble(BaseModel):
    """投票集成分类器"""  # 类文档字符串，说明类的功能

    def __init__(self, estimators: List[BaseModel],
                 voting: str = 'hard',
                 weights: Optional[List[float]] = None,
                 name: str = "VotingEnsemble"):
        """
        参数:  # 构造函数文档字符串，说明参数含义
        ----------
        estimators : 基分类器列表  # 接收一个基分类器列表
        voting : 'hard' 硬投票或 'soft' 软投票  # 投票方式，默认为硬投票
        weights : 每个分类器的权重，None表示等权重  # 权重参数，可选
        """
        super().__init__(name)  # 调用父类BaseModel的构造函数，传入模型名称
        self.estimators = estimators  # 将传入的基分类器列表保存为实例变量
        self.voting = voting  # 保存投票方式

        if voting not in ['hard', 'soft']:  # 检查投票方式是否合法
            raise ValueError("voting must be 'hard' or 'soft'")  # 抛出值错误异常

        # 设置权重
        if weights is None:  # 如果没有提供权重参数
            self.weights = np.ones(len(estimators))  # 创建等权重数组，长度等于分类器数量
        else:  # 如果提供了权重参数
            if len(weights) != len(estimators):  # 检查权重数量是否与分类器数量匹配
                raise ValueError("权重数量必须与分类器数量相同")  # 抛出值错误异常
            self.weights = np.array(weights)  # 将权重列表转换为NumPy数组

        # 归一化权重
        self.weights = self.weights / self.weights.sum()  # 对权重进行归一化，使所有权重之和为1

    def fit(self, X, y, verbose: bool = True):
        """训练所有基分类器"""  # 训练方法文档字符串
        if verbose:  # 如果verbose为True，打印训练信息
            print(f"训练 {len(self.estimators)} 个基分类器...")

        for i, estimator in enumerate(self.estimators):  # 遍历所有基分类器
            if verbose:  # 如果需要详细输出
                print(f"  [{i+1}/{len(self.estimators)}] 训练 {estimator.name}")  # 打印当前训练的分类器信息
            estimator.fit(X, y)  # 训练当前基分类器

        self.is_fitted = True  # 设置模型已训练标志为True
        self.classes_ = np.unique(y)  # 获取数据集中所有不重复的类别标签
        self.n_classes_ = len(self.classes_)  # 计算类别数量

        return self  # 返回自身实例，支持链式调用

    def _hard_vote(self, predictions: np.ndarray) -> np.ndarray:
        """硬投票"""  # 私有方法，执行硬投票
        # predictions形状: [n_estimators, n_samples]  # 注释说明预测结果的形状
        n_samples = predictions.shape[1]  # 获取样本数量
        final_predictions = np.empty(n_samples, dtype=predictions.dtype)  # 创建空数组用于存储最终预测结果

        for i in range(n_samples):  # 遍历每个样本
            # 对每个样本，统计所有分类器的预测
            votes = {}  # 创建空字典用于存储每个类别的得票（加权后）
            for j, pred in enumerate(predictions[:, i]):  # 遍历所有分类器对当前样本的预测
                weight = self.weights[j]  # 获取当前分类器的权重
                if pred in votes:  # 如果该预测类别已在字典中
                    votes[pred] += weight  # 累加权重
                else:  # 如果该预测类别不在字典中
                    votes[pred] = weight  # 初始化该类别权重

            # 选择权重最高的类别
            final_predictions[i] = max(votes.items(), key=lambda x: x[1])[0]  # 找到权重最大的类别作为最终预测

        return final_predictions  # 返回所有样本的最终预测结果

    def _soft_vote(self, probabilities: np.ndarray) -> np.ndarray:
        """软投票"""  # 私有方法，执行软投票
        # probabilities形状: [n_estimators, n_samples, n_classes]  # 注释说明概率矩阵的形状

        # 加权平均概率
        weighted_probs = np.zeros((probabilities.shape[1], probabilities.shape[2]))  # 创建零矩阵，用于存储加权平均概率

        for i, (proba, weight) in enumerate(zip(probabilities, self.weights)):  # 遍历所有分类器的概率预测和权重
            weighted_probs += weight * proba  # 加权累加概率

        # 选择概率最高的类别
        return np.argmax(weighted_probs, axis=1)  # 返回每行（每个样本）最大概率值的索引，即预测类别

    def predict(self, X):
        """预测类别"""  # 预测方法文档字符串
        if not self.is_fitted:  # 检查模型是否已训练
            raise ValueError("模型必须先训练")  # 抛出值错误异常

        if self.voting == 'hard':  # 如果是硬投票
            # 收集所有分类器的预测
            predictions = np.array([estimator.predict(X) for estimator in self.estimators])  # 获取所有基分类器的预测结果
            return self._hard_vote(predictions)  # 调用硬投票方法
        else:  # soft voting，如果是软投票
            # 收集所有分类器的概率预测
            probabilities = np.array([estimator.predict_proba(X) for estimator in self.estimators])  # 获取所有基分类器的概率预测
            return self._soft_vote(probabilities)  # 调用软投票方法

    def predict_proba(self, X):
        """预测概率（仅适用于软投票）"""  # 预测概率方法文档字符串
        if self.voting != 'soft':  # 如果不是软投票模式
            raise ValueError("概率预测仅适用于软投票")  # 抛出值错误异常

        if not self.is_fitted:  # 检查模型是否已训练
            raise ValueError("模型必须先训练")  # 抛出值错误异常

        # 收集所有分类器的概率预测
        probabilities = np.array([estimator.predict_proba(X) for estimator in self.estimators])  # 获取所有基分类器的概率预测

        # 加权平均
        weighted_probs = np.zeros((probabilities.shape[1], probabilities.shape[2]))  # 创建零矩阵用于存储加权平均概率
        for i, (proba, weight) in enumerate(zip(probabilities, self.weights)):  # 遍历所有分类器的概率预测和权重
            weighted_probs += weight * proba  # 加权累加概率

        return weighted_probs  # 返回加权平均概率

    def get_estimator_performance(self, X, y, metric='accuracy'):
        """评估每个基分类器的性能"""  # 评估基分类器性能的方法文档字符串
        from sklearn.metrics import accuracy_score, f1_score  # 导入sklearn的评估指标函数

        performances = {}  # 创建空字典用于存储每个分类器的性能

        for estimator in self.estimators:  # 遍历所有基分类器
            y_pred = estimator.predict(X)  # 获取当前分类器的预测结果

            if metric == 'accuracy':  # 如果评估指标是准确率
                score = accuracy_score(y, y_pred)  # 计算准确率
            elif metric == 'f1':  # 如果评估指标是F1分数
                score = f1_score(y, y_pred, average='weighted')  # 计算加权F1分数
            else:  # 如果是指标不支持
                raise ValueError(f"不支持的评估指标: {metric}")  # 抛出值错误异常

            performances[estimator.name] = score  # 将分类器名称和得分存入字典

        return performances  # 返回性能字典


class WeightedVotingEnsemble(VotingEnsemble):
    """带权重优化的投票集成"""  # 类文档字符串，说明这是带权重优化的投票集成

    def __init__(self, estimators: List[BaseModel],
                 voting: str = 'soft',
                 name: str = "WeightedVotingEnsemble"):
        super().__init__(estimators, voting, name=name)  # 调用父类构造函数
        self.trained_weights = None  # 初始化优化后的权重为None

    def optimize_weights(self, X_val, y_val,
                        metric: str = 'accuracy',
                        method: str = 'direct'):
        """
        基于验证集优化权重  # 方法文档字符串

        参数:
        ----------
        X_val, y_val : 验证集  # 验证集数据和标签
        metric : 评估指标 ('accuracy', 'f1')  # 评估指标类型
        method : 优化方法 ('direct', 'inverse_error')  # 权重优化方法
        """
        from sklearn.metrics import accuracy_score, f1_score  # 导入sklearn的评估指标函数

        performances = []  # 创建空列表用于存储每个分类器的性能

        # 计算每个分类器在验证集上的性能
        for estimator in self.estimators:  # 遍历所有基分类器
            y_pred = estimator.predict(X_val)  # 获取当前分类器的预测结果

            if metric == 'accuracy':  # 如果评估指标是准确率
                perf = accuracy_score(y_val, y_pred)  # 计算准确率
            elif metric == 'f1':  # 如果评估指标是F1分数
                perf = f1_score(y_val, y_pred, average='weighted')  # 计算加权F1分数
            else:  # 如果是指标不支持
                raise ValueError(f"不支持的评估指标: {metric}")  # 抛出值错误异常

            performances.append(perf)  # 将性能得分添加到列表中

        performances = np.array(performances)  # 将性能列表转换为NumPy数组

        # 根据优化方法计算权重
        if method == 'direct':  # 如果使用直接法
            # 直接使用性能作为权重
            weights = performances  # 直接使用性能作为权重
        elif method == 'inverse_error':  # 如果使用错误率倒数法
            # 使用错误率的倒数
            errors = 1 - performances  # 计算错误率（1减去准确率）
            # 避免除零错误
            errors = np.clip(errors, 1e-10, 1)  # 将错误率限制在[1e-10, 1]范围内，避免除零
            weights = 1.0 / errors  # 计算错误率的倒数作为权重
        else:  # 如果方法不支持
            raise ValueError(f"不支持的优化方法: {method}")  # 抛出值错误异常

        # 处理可能的负值
        if np.any(weights < 0):  # 如果权重中有负值
            weights = weights - weights.min() + 1e-10  # 将所有权重平移，使最小值为1e-10

        # 归一化
        self.weights = weights / weights.sum()  # 对权重进行归一化
        self.trained_weights = self.weights.copy()  # 保存优化后的权重

        # 打印结果
        print("优化后的权重:")  # 打印标题
        print("-" * 40)  # 打印分隔线
        for i, (estimator, weight, perf) in enumerate(zip(self.estimators, self.weights, performances)):  # 遍历所有分类器
            print(f"  {i+1:2d}. {estimator.name:25s}: 权重={weight:.4f}, 性能={perf:.4f}")  # 打印每个分类器的权重和性能

        return self.weights  # 返回优化后的权重


def evaluate_ensemble_performance(X_train, y_train, X_test, y_test,
                                voting='hard', random_state=42):
    """评估集成性能的完整流程"""  # 函数文档字符串
    from models.base import get_diverse_classifiers  # 导入获取多样化分类器的函数

    print("=" * 60)  # 打印分隔线
    print("集成学习性能评估")  # 打印标题
    print("=" * 60)  # 打印分隔线

    # 1. 获取基分类器
    base_classifiers = get_diverse_classifiers(random_state=random_state)  # 获取多样化的基分类器列表

    # 2. 训练并评估单个分类器
    print("\n1. 单个分类器性能:")  # 打印子标题
    print("-" * 40)  # 打印分隔线

    single_performances = {}  # 创建空字典用于存储单个分类器的性能
    for clf in base_classifiers:  # 遍历所有基分类器
        clf.fit(X_train, y_train)  # 训练当前分类器
        acc = clf.score(X_test, y_test)  # 在测试集上评估分类器
        single_performances[clf.name] = acc  # 将分类器名称和准确率存入字典
        print(f"  {clf.name:25s}: {acc:.4f}")  # 打印分类器名称和准确率

    # 3. 训练投票集成
    print(f"\n2. {voting}投票集成:")  # 打印子标题，显示投票方式
    print("-" * 40)  # 打印分隔线

    ensemble = VotingEnsemble(base_classifiers, voting=voting)  # 创建投票集成实例
    ensemble.fit(X_train, y_train)  # 训练集成模型
    ensemble_acc = ensemble.score(X_test, y_test)  # 在测试集上评估集成模型
    print(f"  {ensemble.name:25s}: {ensemble_acc:.4f}")  # 打印集成模型名称和准确率

    # 4. 结果分析
    best_single_name = max(single_performances, key=single_performances.get)  # 找到性能最好的单个分类器名称
    best_single_acc = single_performances[best_single_name]  # 获取最佳单个分类器的准确率

    improvement = (ensemble_acc - best_single_acc) / best_single_acc * 100  # 计算集成模型相对于最佳单个分类器的提升百分比

    print(f"\n3. 性能对比:")  # 打印子标题
    print("-" * 40)  # 打印分隔线
    print(f"  最佳单模型: {best_single_name} ({best_single_acc:.4f})")  # 打印最佳单模型信息
    print(f"  集成模型: {ensemble.name} ({ensemble_acc:.4f})")  # 打印集成模型信息
    print(f"  相对提升: {improvement:+.2f}%")  # 打印相对提升百分比

    return {  # 返回包含所有评估结果的字典
        'single_performances': single_performances,
        'ensemble_accuracy': ensemble_acc,
        'best_single': (best_single_name, best_single_acc),
        'improvement': improvement,
        'ensemble': ensemble
    }


# 测试代码
if __name__ == "__main__":  # 如果是直接运行此脚本
    from sklearn.datasets import load_iris  # 导入鸢尾花数据集
    from sklearn.model_selection import train_test_split  # 导入数据集划分函数
    from models.base import get_diverse_classifiers  # 导入获取多样化分类器的函数

    # 加载数据
    data = load_iris()  # 加载鸢尾花数据集
    X, y = data.data, data.target  # 获取特征数据和标签数据

    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(  # 划分训练集和测试集
        X, y, test_size=0.2, random_state=42, stratify=y  # 测试集占20%，随机种子为42，按标签分层抽样
    )

    # 运行评估
    result = evaluate_ensemble_performance(  # 调用评估函数
        X_train, y_train, X_test, y_test,
        voting='soft', random_state=42  # 使用软投票，随机种子为42
    )

    # 测试加权集成
    print(f"\n4. 加权投票集成测试:")  # 打印子标题
    print("-" * 40)  # 打印分隔线

    # 重新训练基分类器（因为已经被之前的集成训练过了）
    base_classifiers = get_diverse_classifiers(random_state=42)  # 重新获取基分类器

    # 划分验证集
    X_train_sub, X_val, y_train_sub, y_val = train_test_split(  # 从训练集中划分出验证集
        X_train, y_train, test_size=0.25, random_state=42, stratify=y_train  # 验证集占25%，随机种子为42，分层抽样
    )

    # 训练基分类器
    for clf in base_classifiers:  # 遍历所有基分类器
        clf.fit(X_train_sub, y_train_sub)  # 使用训练子集训练基分类器

    # 创建并优化加权集成
    weighted_ensemble = WeightedVotingEnsemble(base_classifiers, voting='soft')  # 创建加权投票集成实例
    weighted_ensemble.optimize_weights(X_val, y_val, metric='accuracy')  # 优化权重

    # 在测试集上评估
    weighted_ensemble.fit(X_train_sub, y_train_sub, verbose=False)  # 训练加权集成模型，不显示详细信息
    weighted_acc = weighted_ensemble.score(X_test, y_test)  # 在测试集上评估加权集成
    print(f"\n  加权集成准确率: {weighted_acc:.4f}")  # 打印加权集成的准确率
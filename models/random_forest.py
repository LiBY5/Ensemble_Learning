"""随机森林实现 - models/random_forest.py """

import numpy as np  # 导入NumPy库，用于数值计算
from copy import deepcopy  # 导入深拷贝函数
from typing import Optional, Union  # 导入类型提示相关模块
from sklearn.tree import DecisionTreeClassifier  # 导入sklearn的决策树分类器
from sklearn.metrics import accuracy_score  # 导入准确率评估指标
import warnings  # 导入警告模块

from models.base import BaseClassifier  # 从自定义模块导入基础分类器基类


class RandomForestClassifier(BaseClassifier):
    """随机森林分类器"""

    # 初始化方法，定义随机森林的所有超参数
    def __init__(self,
                 n_estimators: int = 100,  # 森林中树的数量，默认100
                 criterion: str = 'gini',  # 分裂标准，默认基尼指数
                 max_depth: Optional[int] = None,  # 树的最大深度，None表示不限制
                 min_samples_split: Union[int, float] = 2,  # 内部节点分裂所需最小样本数
                 min_samples_leaf: Union[int, float] = 1,  # 叶节点最少样本数
                 max_features: Union[str, float] = 'sqrt',  # 分裂时考虑的特征数
                 bootstrap: bool = True,  # 是否使用bootstrap采样
                 oob_score: bool = False,  # 是否计算袋外分数
                 random_state: Optional[int] = None,  # 随机种子
                 n_jobs: Optional[int] = None,  # 并行作业数（此处未使用，为API兼容性保留）
                 verbose: int = 0):  # 日志详细程度
        super().__init__(name="RandomForestClassifier")  # 调用父类初始化方法

        # 树的数量和配置
        self.n_estimators = n_estimators  # 设置树的数量
        self.criterion = criterion  # 设置分裂标准
        self.max_depth = max_depth  # 设置最大深度
        self.min_samples_split = min_samples_split  # 设置分裂最小样本数
        self.min_samples_leaf = min_samples_leaf  # 设置叶节点最小样本数
        self.max_features = max_features  # 设置最大特征数

        # Bagging相关参数
        self.bootstrap = bootstrap  # 设置是否使用bootstrap采样
        self.oob_score = oob_score  # 设置是否计算袋外分数

        # 并行化和随机性
        self.random_state = random_state  # 设置随机种子
        self.n_jobs = n_jobs  # 设置并行作业数（保留参数）
        self.verbose = verbose  # 设置日志详细程度

        # 内部状态（将在fit方法中初始化）
        self.estimators_ = []  # 存储所有决策树模型
        self.estimator_features_ = []  # 记录每棵树使用的特征索引
        self.feature_importances_ = None  # 特征重要性数组
        self.oob_score_ = None  # 袋外分数

        # 设置随机种子
        if random_state is not None:  # 如果提供了随机种子
            np.random.seed(random_state)  # 设置NumPy的随机种子

    def fit(self, X, y):
        """训练随机森林"""
        # 数据验证
        X, y = self._validate_data(X, y)  # 验证和转换输入数据
        n_samples, n_features = X.shape  # 获取样本数和特征数

        # 计算实际的最大特征数（用于特征采样）
        max_features = self._compute_max_features(n_features)  # 根据参数计算每棵树使用的特征数

        # 初始化OOB估计（袋外样本预测统计）
        oob_pred_sum = None  # 存储每个样本的各类别预测概率和
        oob_count = None  # 存储每个样本被预测的次数

        if self.oob_score:  # 如果需要计算袋外分数
            n_classes = len(np.unique(y))  # 获取类别数
            oob_pred_sum = np.zeros((n_samples, n_classes))  # 初始化预测概率和矩阵
            oob_count = np.zeros(n_samples)  # 初始化预测计数数组

        # 清空之前的模型（确保每次fit都是重新训练）
        self.estimators_ = []  # 清空树列表
        self.estimator_features_ = []  # 清空特征索引列表

        # 训练每棵树
        for i in range(self.n_estimators):  # 遍历每棵树
            if self.verbose > 0 and i % 10 == 0:  # 如果设置了详细输出且是10的倍数
                print(f"训练决策树 {i+1}/{self.n_estimators}")  # 打印训练进度

            # 1. 样本采样（bootstrap采样）
            if self.bootstrap:  # 如果使用bootstrap采样
                indices = np.random.choice(n_samples, size=n_samples, replace=True)  # 有放回随机采样
                X_sample = X[indices]  # 获取采样后的特征数据
                y_sample = y[indices]  # 获取采样后的标签数据
                sample_indices = indices  # 记录采样索引（用于OOB计算）
            else:  # 如果不使用bootstrap采样
                X_sample, y_sample = X, y  # 使用所有数据
                sample_indices = np.arange(n_samples)  # 使用所有索引

            # 2. 特征采样（随机森林的关键 - 随机特征子空间）
            feature_indices = np.random.choice(
                n_features,  # 从所有特征中
                size=max_features,  # 选择max_features个
                replace=False  # 无放回采样
            )
            self.estimator_features_.append(feature_indices)  # 记录当前树使用的特征索引

            # 3. 训练决策树
            tree = DecisionTreeClassifier(  # 创建决策树分类器
                criterion=self.criterion,  # 设置分裂标准
                max_depth=self.max_depth,  # 设置最大深度
                min_samples_split=self.min_samples_split,  # 设置分裂最小样本数
                min_samples_leaf=self.min_samples_leaf,  # 设置叶节点最小样本数
                random_state=self.random_state + i if self.random_state is not None else None  # 为每棵树设置不同随机种子
            )

            # 使用特征子集训练决策树
            tree.fit(X_sample[:, feature_indices], y_sample)  # 只使用选中的特征进行训练
            self.estimators_.append(tree)  # 将训练好的树添加到列表中

            # 4. 更新OOB估计（用于计算袋外分数）
            if self.oob_score and self.bootstrap:  # 如果需要计算袋外分数且使用了bootstrap采样
                self._update_oob_estimation(  # 调用更新OOB估计的方法
                    i, X, y, sample_indices, feature_indices,
                    oob_pred_sum, oob_count
                )

        # 5. 计算特征重要性（基于所有树的特征重要性）
        self.feature_importances_ = self._compute_feature_importances(n_features)

        # 6. 完成训练（设置最终状态）
        self._finalize_fit(X, y, oob_pred_sum, oob_count)

        return self  # 返回self以支持链式调用

    def predict(self, X):
        """预测类别 - 多数投票"""
        if not self.is_fitted:  # 如果模型尚未训练
            raise ValueError("模型必须先训练")  # 抛出异常

        # 收集所有树的预测
        all_predictions = []  # 初始化预测列表

        for tree, feature_indices in zip(self.estimators_, self.estimator_features_):  # 遍历每棵树及其特征索引
            # 使用特征子集进行预测
            X_selected = X[:, feature_indices]  # 选择当前树使用的特征
            pred = tree.predict(X_selected)  # 使用当前树进行预测
            all_predictions.append(pred)  # 将预测结果添加到列表

        # 转为数组：[n_trees, n_samples]
        all_predictions = np.array(all_predictions)  # 将预测列表转换为NumPy数组

        # 多数投票（集成学习的关键步骤）
        final_predictions = []  # 初始化最终预测列表
        for sample_idx in range(all_predictions.shape[1]):  # 遍历每个样本
            counts = np.bincount(all_predictions[:, sample_idx], minlength=self.n_classes_)  # 统计各类别得票数
            final_predictions.append(np.argmax(counts))  # 选择得票最多的类别

        return np.array(final_predictions)  # 返回最终预测结果数组

    def predict_proba(self, X):
        """预测概率 - 平均概率"""
        if not self.is_fitted:  # 如果模型尚未训练
            raise ValueError("模型必须先训练")  # 抛出异常

        # 收集所有树的概率预测
        all_probas = []  # 初始化概率列表

        for tree, feature_indices in zip(self.estimators_, self.estimator_features_):  # 遍历每棵树及其特征索引
            X_selected = X[:, feature_indices]  # 选择当前树使用的特征
            proba = tree.predict_proba(X_selected)  # 获取当前树的预测概率
            all_probas.append(proba)  # 将概率添加到列表

        # 平均所有树的概率
        avg_proba = np.mean(all_probas, axis=0)  # 沿第一个维度（树的数量）求平均
        return avg_proba  # 返回平均概率

    def _validate_data(self, X, y):
        """数据验证"""
        X = np.array(X)  # 将X转换为NumPy数组
        y = np.array(y)  # 将y转换为NumPy数组

        if len(X) != len(y):  # 检查X和y的长度是否一致
            raise ValueError(f"X和y的长度不匹配: {len(X)} != {len(y)}")  # 抛出异常

        return X, y  # 返回转换后的数据

    def _compute_max_features(self, n_features):
        """计算每次分裂考虑的特征数"""
        if isinstance(self.max_features, str):  # 如果max_features是字符串类型
            if self.max_features == 'auto':  # 如果是'auto'（传统用法）
                return int(np.sqrt(n_features))  # 返回sqrt(n_features)
            elif self.max_features == 'sqrt':  # 如果是'sqrt'
                return int(np.sqrt(n_features))  # 返回sqrt(n_features)
            elif self.max_features == 'log2':  # 如果是'log2'
                return int(np.log2(n_features))  # 返回log2(n_features)
            else:
                raise ValueError(f"不支持的max_features字符串: {self.max_features}")  # 抛出异常
        elif isinstance(self.max_features, float):  # 如果max_features是浮点数类型
            if self.max_features <= 0.0 or self.max_features > 1.0:  # 检查范围
                raise ValueError("max_features比例必须在(0, 1]范围内")  # 抛出异常
            return max(1, int(self.max_features * n_features))  # 返回比例乘特征数，至少为1
        elif isinstance(self.max_features, int):  # 如果max_features是整数类型
            if self.max_features <= 0:  # 检查是否为正
                raise ValueError("max_features必须为正整数")  # 抛出异常
            return min(self.max_features, n_features)  # 返回不超过特征数的值
        else:
            raise TypeError("max_features必须是字符串、浮点数或整数")  # 抛出类型错误异常

    def _update_oob_estimation(self, tree_idx, X, y, sample_indices,
                             feature_indices, oob_pred_sum, oob_count):
        """更新OOB估计"""
        n_samples = X.shape[0]  # 获取样本总数

        # 找出OOB样本（不在当前树训练集中的样本）
        all_indices = set(range(n_samples))  # 所有样本索引的集合
        in_bag_indices = set(sample_indices)  # 当前树训练集索引的集合
        oob_indices = list(all_indices - in_bag_indices)  # OOB样本索引列表

        if len(oob_indices) == 0:  # 如果没有OOB样本
            return  # 直接返回

        # 用当前树预测OOB样本
        tree = self.estimators_[tree_idx]  # 获取当前树
        X_oob = X[oob_indices]  # 获取OOB样本的特征
        X_oob_selected = X_oob[:, feature_indices]  # 选择当前树使用的特征
        y_oob_pred_proba = tree.predict_proba(X_oob_selected)  # 预测OOB样本的概率

        # 累加预测（为后续计算平均概率做准备）
        for idx, pred in zip(oob_indices, y_oob_pred_proba):  # 遍历每个OOB样本及其预测
            oob_pred_sum[idx] += pred  # 累加预测概率
            oob_count[idx] += 1  # 增加预测次数计数

    def _compute_feature_importances(self, n_features):
        """计算基尼重要性"""
        importances = np.zeros(n_features)  # 初始化特征重要性数组

        for tree, feature_indices in zip(self.estimators_, self.estimator_features_):  # 遍历每棵树及其特征索引
            # 获取树的特征重要性（基于基尼不纯度减少）
            tree_importances = tree.feature_importances_

            # 将重要性值加到对应的原始特征位置上
            for idx, importance in zip(feature_indices, tree_importances):  # 遍历特征索引及其重要性
                importances[idx] += importance  # 累加重要性

        # 归一化（使所有特征重要性之和为1）
        if importances.sum() > 0:  # 如果总和大于0
            importances /= importances.sum()  # 归一化

        return importances  # 返回特征重要性数组

    def _finalize_fit(self, X, y, oob_pred_sum=None, oob_count=None):
        """完成训练"""
        self.is_fitted = True  # 设置模型已训练标志
        self.classes_ = np.unique(y)  # 存储所有类别标签
        self.n_classes_ = len(self.classes_)  # 存储类别数
        self.n_features_ = X.shape[1]  # 存储特征数

        # 计算OOB分数
        if self.oob_score and oob_pred_sum is not None and oob_count is not None:  # 如果需要计算OOB分数且有数据
            valid_oob = oob_count > 0  # 找出至少被一棵树预测过的样本
            if np.any(valid_oob):  # 如果有有效的OOB样本
                oob_decision = oob_pred_sum[valid_oob] / oob_count[valid_oob, np.newaxis]  # 计算平均概率
                y_oob_pred = np.argmax(oob_decision, axis=1)  # 选择概率最大的类别作为预测
                y_oob_true = y[valid_oob]  # 获取OOB样本的真实标签
                self.oob_score_ = accuracy_score(y_oob_true, y_oob_pred)  # 计算袋外准确率

                if self.verbose > 0:  # 如果设置了详细输出
                    print(f"OOB分数: {self.oob_score_:.4f} (基于{np.sum(valid_oob)}个样本)")  # 打印OOB分数
            else:  # 如果没有有效的OOB样本
                warnings.warn("没有有效的OOB样本，无法计算OOB分数")  # 发出警告
                self.oob_score_ = None  # 设置OOB分数为None
        else:  # 如果不计算OOB分数
            self.oob_score_ = None  # 设置OOB分数为None
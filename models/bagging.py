"""Bagging与随机森林实现"""
import numpy as np
from copy import deepcopy
from typing import List, Optional, Union, Tuple
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import resample
from sklearn.metrics import accuracy_score
import warnings

from models.base import BaseClassifier  # 导入自定义基类


class BaggingClassifier(BaseClassifier):
    """Bagging分类器 - 集成学习方法，通过自助采样和特征采样创建多个基学习器"""

    def __init__(self,
                 base_estimator: BaseEstimator,  # 基学习器，可以是任何sklearn兼容的分类器
                 n_estimators: int = 10,  # 集成中基学习器的数量
                 max_samples: float = 1.0,  # 每个基学习器使用的样本比例
                 max_features: float = 1.0,  # 每个基学习器使用的特征比例
                 bootstrap: bool = True,  # 是否使用自助采样（有放回）
                 bootstrap_features: bool = False,  # 是否对特征进行自助采样
                 oob_score: bool = False,  # 是否计算袋外（Out-of-Bag）分数
                 random_state: Optional[int] = None,  # 随机种子，确保结果可重现
                 verbose: int = 0):  # 控制训练过程中的输出详细程度
        """
        参数初始化 - 仔细理解每个参数
        """
        super().__init__(name="BaggingClassifier")  # 调用父类初始化方法，设置分类器名称

        # 存储所有传入参数到实例属性
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.bootstrap_features = bootstrap_features
        self.oob_score = oob_score
        self.random_state = random_state
        self.verbose = verbose

        # 初始化存储结构
        self.estimators_ = []  # 存储所有训练好的基学习器
        self.estimator_features_ = []  # 存储每个基学习器使用的特征索引
        self.oob_decision_function_ = None  # 存储OOB样本的预测概率
        self.oob_score_ = None  # 存储OOB准确率分数

        # 设置随机种子以确保结果可重现
        if random_state is not None:
            np.random.seed(random_state)

    def fit(self, X, y):
        """训练Bagging分类器 - 核心方法"""
        # 输入验证：确保数据格式正确
        X, y = self._validate_data(X, y)

        n_samples, n_features = X.shape  # 获取样本数和特征数
        n_subsample = int(n_samples * self.max_samples)  # 计算每个基学习器的样本数

        # 初始化变量用于OOB估计
        oob_pred_sum = None  # 存储OOB样本的预测概率总和
        oob_count = None  # 存储每个OOB样本被预测的次数

        # 如果启用OOB评分，初始化OOB数据结构
        if self.oob_score:
            n_classes = len(np.unique(y))  # 获取类别数
            oob_pred_sum = np.zeros((n_samples, n_classes))  # 初始化概率累加矩阵
            oob_count = np.zeros(n_samples)  # 初始化计数数组

        # 清空之前训练的结果
        self.estimators_ = []
        self.estimator_features_ = []

        # 训练n_estimators个基学习器
        for i in range(self.n_estimators):
            # 如果verbose大于0，每10个基学习器输出一次进度
            if self.verbose > 0 and i % 10 == 0:
                print(f"训练基学习器 {i + 1}/{self.n_estimators}")

            # 1. 样本采样：从原始数据集中采样训练子集
            X_sample, y_sample, sample_indices = self._bootstrap_sample(X, y, n_subsample)

            # 2. 特征采样：从特征中采样特征子集
            X_sample_selected, feature_indices = self._feature_sample(X_sample)
            self.estimator_features_.append(feature_indices)  # 存储特征索引

            # 3. 训练基学习器 - 使用深拷贝创建独立的学习器实例
            estimator = deepcopy(self.base_estimator)
            estimator.fit(X_sample_selected, y_sample)  # 在采样子集上训练
            self.estimators_.append(estimator)  # 存储训练好的学习器

            # 4. 更新OOB估计：用当前学习器预测OOB样本
            if self.oob_score:
                self._update_oob_estimation(i, X, sample_indices, feature_indices,
                                            oob_pred_sum, oob_count)

        # 最终处理：设置训练完成标志，计算OOB分数等
        self._finalize_fit(X, y, oob_pred_sum, oob_count)
        return self  # 返回self以支持链式调用

    def predict(self, X):
        """预测类别 - 多数投票"""
        if not self.is_fitted:  # 检查模型是否已训练
            raise ValueError("请先调用 fit() 方法训练模型")

        # 获取所有基学习器的预测
        predictions = []
        for estimator, feature_indices in zip(self.estimators_, self.estimator_features_):
            X_selected = X[:, feature_indices]  # 选择当前学习器使用的特征
            pred = estimator.predict(X_selected)  # 获取预测结果
            predictions.append(pred)

        # 多数投票：对每个样本，统计所有学习器的预测结果，选择票数最多的类别
        predictions = np.array(predictions)  # 转换为数组，形状为[n_estimators, n_samples]
        final_predictions = []  # 存储最终预测结果

        for sample_idx in range(predictions.shape[1]):  # 遍历每个样本
            # 统计当前样本的各类别票数
            counts = np.bincount(predictions[:, sample_idx], minlength=self.n_classes_)
            final_predictions.append(np.argmax(counts))  # 选择票数最多的类别

        return np.array(final_predictions)  # 返回最终预测数组

    def predict_proba(self, X):
        """预测概率 - 平均概率"""
        if not self.is_fitted:  # 检查模型是否已训练
            raise ValueError("请先调用 fit() 方法训练模型")

        probas = []  # 存储所有基学习器的概率预测
        for estimator, feature_indices in zip(self.estimators_, self.estimator_features_):
            X_selected = X[:, feature_indices]  # 选择特征
            proba = estimator.predict_proba(X_selected)  # 获取概率预测
            probas.append(proba)

        # 平均所有基学习器的概率：对每个样本的每个类别取平均
        avg_proba = np.mean(probas, axis=0)  # 沿基学习器维度求平均
        return avg_proba  # 返回平均概率

    # ===================== 辅助方法 =====================

    def _validate_data(self, X, y):
        """数据验证 - 确保输入数据的有效性"""
        X = np.array(X)  # 转换为numpy数组
        y = np.array(y)  # 转换为numpy数组

        if len(X) != len(y):  # 检查样本数是否一致
            raise ValueError(f"X和y的长度不匹配: {len(X)} != {len(y)}")

        return X, y  # 返回验证后的数据

    def _bootstrap_sample(self, X, y, n_samples):
        """自助采样 - 根据bootstrap参数决定采样方式"""
        n_total = len(X)  # 总样本数

        if self.bootstrap:
            # 有放回采样：允许样本重复出现
            indices = np.random.choice(n_total, size=n_samples, replace=True)
        else:
            # 无放回采样：样本不重复
            indices = np.random.choice(n_total, size=n_samples, replace=False)

        # 返回采样的数据、标签和对应的索引
        return X[indices], y[indices], indices

    def _feature_sample(self, X):
        """特征采样 - 根据bootstrap_features参数决定采样方式"""
        n_features = X.shape[1]  # 特征总数
        n_selected = int(n_features * self.max_features)  # 需要选择的特征数

        if n_selected == 0:  # 防止选择0个特征
            raise ValueError("max_features太小，至少选择一个特征")

        if self.bootstrap_features:
            # 有放回特征采样：允许特征重复出现
            feature_indices = np.random.choice(n_features, size=n_selected, replace=True)
        else:
            # 无放回特征采样：特征不重复
            feature_indices = np.random.choice(n_features, size=n_selected, replace=False)

        # 返回选择后的特征子集和对应的特征索引
        return X[:, feature_indices], feature_indices

    def _update_oob_estimation(self, estimator_idx, X, sample_indices,
                               feature_indices, oob_pred_sum, oob_count):
        """更新OOB估计 - 使用当前基学习器预测OOB样本"""
        n_samples = X.shape[0]  # 总样本数

        # 找出OOB样本：不在当前采样袋中的样本
        all_indices = set(range(n_samples))  # 所有样本索引
        in_bag_indices = set(sample_indices)  # 袋内样本索引
        oob_indices = list(all_indices - in_bag_indices)  # OOB样本索引

        if len(oob_indices) == 0:  # 如果没有OOB样本，直接返回
            return

        # 用当前基学习器预测OOB样本
        estimator = self.estimators_[estimator_idx]  # 获取当前学习器
        X_oob = X[oob_indices]  # 获取OOB样本
        X_oob_selected = X_oob[:, feature_indices]  # 选择对应的特征

        # 检查基学习器是否有predict_proba方法
        if hasattr(estimator, 'predict_proba'):
            # 如果有，直接获取概率预测
            y_oob_pred = estimator.predict_proba(X_oob_selected)
        else:
            # 如果没有，使用predict并转换为one-hot编码
            y_pred = estimator.predict(X_oob_selected)  # 获取类别预测
            y_oob_pred = np.zeros((len(y_pred), self.n_classes_))  # 初始化概率矩阵
            for i, cls in enumerate(y_pred):  # 遍历每个预测
                y_oob_pred[i, cls] = 1.0  # 将对应类别设为1.0

        # 累加预测：更新OOB样本的预测概率和计数
        for idx, pred in zip(oob_indices, y_oob_pred):
            oob_pred_sum[idx] += pred  # 累加概率
            oob_count[idx] += 1  # 增加计数

    def _finalize_fit(self, X, y, oob_pred_sum=None, oob_count=None):
        """完成训练 - 设置模型状态和计算OOB分数"""
        self.is_fitted = True  # 设置模型已训练标志
        self.classes_ = np.unique(y)  # 存储所有唯一类别
        self.n_classes_ = len(self.classes_)  # 存储类别数
        self.n_features_ = X.shape[1]  # 存储特征数

        # 计算OOB分数（如果启用）
        if self.oob_score and oob_pred_sum is not None and oob_count is not None:
            valid_oob = oob_count > 0  # 找到有OOB预测的样本
            if np.any(valid_oob):  # 如果有有效的OOB样本
                # 计算平均概率：总概率/计数
                self.oob_decision_function_ = oob_pred_sum[valid_oob] / oob_count[valid_oob, np.newaxis]
                y_oob_pred = np.argmax(self.oob_decision_function_, axis=1)  # 获取预测类别
                y_oob_true = y[valid_oob]  # 获取真实标签
                self.oob_score_ = accuracy_score(y_oob_true, y_oob_pred)  # 计算准确率

                if self.verbose > 0:  # 如果启用详细输出
                    print(f"OOB分数: {self.oob_score_:.4f} (基于{np.sum(valid_oob)}个样本)")
            else:
                # 如果没有有效的OOB样本，发出警告
                warnings.warn("没有有效的OOB样本，无法计算OOB分数")
                self.oob_score_ = None

    def get_feature_importances(self):
        """获取特征重要性（Bagging没有天然的特征重要性，返回None）"""
        warnings.warn("BaggingClassifier没有内置的特征重要性计算方法")
        return None  # 返回None，因为标准Bagging不提供特征重要性
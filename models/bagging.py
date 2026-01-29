"""Bagging与随机森林实现 - models/bagging.py """
import numpy as np  # 导入NumPy数值计算库
from copy import deepcopy  # 导入深拷贝函数
from typing import List, Optional, Union, Tuple  # 导入类型提示
from sklearn.base import BaseEstimator, ClassifierMixin  # 导入sklearn基类
from sklearn.utils import resample  # 导入重采样工具
from sklearn.metrics import accuracy_score  # 导入准确率计算
import warnings  # 导入警告处理

from models.base import BaseClassifier  # 导入基础分类器


class BaggingClassifier(BaseClassifier):
    """Bagging分类器（Bootstrap Aggregating）"""

    def __init__(self,
                 base_estimator: BaseEstimator,  # 基学习器
                 n_estimators: int = 10,         # 基学习器数量
                 max_samples: float = 1.0,       # 每个基学习器使用的样本比例
                 max_features: float = 1.0,      # 每个基学习器使用的特征比例
                 bootstrap: bool = True,         # 是否对样本进行有放回采样
                 bootstrap_features: bool = False,  # 是否对特征进行有放回采样
                 oob_score: bool = False,        # 是否计算袋外分数
                 random_state: Optional[int] = None,  # 随机种子
                 verbose: int = 0):              # 详细程度
        """
        参数初始化
        """
        super().__init__(name="BaggingClassifier")  # 调用父类初始化
        self.base_estimator = base_estimator  # 基学习器
        self.n_estimators = n_estimators      # 基学习器数量
        self.max_samples = max_samples        # 样本采样比例
        self.max_features = max_features      # 特征采样比例
        self.bootstrap = bootstrap            # 是否使用自助采样
        self.bootstrap_features = bootstrap_features  # 特征采样方式
        self.oob_score = oob_score            # 是否计算OOB分数
        self.random_state = random_state      # 随机种子
        self.verbose = verbose                # 详细程度

        # 初始化存储结构
        self.estimators_ = []  # 存储所有基学习器实例
        self.estimator_features_ = []  # 存储每个基学习器使用的特征索引
        self.oob_decision_function_ = None  # 存储OOB决策函数（软预测）
        self.oob_score_ = None  # 存储OOB分数

        # 设置随机种子
        if random_state is not None:
            np.random.seed(random_state)  # 设置全局随机种子

    def get_params(self, deep=True):
        """获取模型参数 - sklearn兼容性"""
        return {
            'base_estimator': self.base_estimator,
            'n_estimators': self.n_estimators,
            'max_samples': self.max_samples,
            'max_features': self.max_features,
            'bootstrap': self.bootstrap,
            'bootstrap_features': self.bootstrap_features,
            'oob_score': self.oob_score,
            'random_state': self.random_state,
            'verbose': self.verbose
        }

    def set_params(self, **params):
        """设置模型参数 - sklearn兼容性"""
        for key, value in params.items():  # 遍历传入的参数键值对
            setattr(self, key, value)       # 动态设置对象属性
        return self  # 返回对象自身以支持链式调用

    def fit(self, X, y):
        """训练Bagging分类器"""
        # 输入验证
        X, y = self._validate_data(X, y)  # 验证并转换数据格式

        n_samples, n_features = X.shape  # 获取样本数和特征数
        n_subsample = int(n_samples * self.max_samples)  # 计算每个基学习器的样本数

        # 初始化OOB数据结构
        oob_pred_sum = None  # OOB预测累加器
        oob_count = None     # OOB样本预测次数计数器

        if self.oob_score:  # 如果需要计算OOB分数
            n_classes = len(np.unique(y))  # 获取类别数
            oob_pred_sum = np.zeros((n_samples, n_classes))  # 初始化概率累加矩阵
            oob_count = np.zeros(n_samples)  # 初始化预测次数向量

        # 清空之前的结果
        self.estimators_ = []  # 清空基学习器列表
        self.estimator_features_ = []  # 清空特征索引列表

        # 训练n_estimators个基学习器
        for i in range(self.n_estimators):
            # 打印训练进度
            if self.verbose > 0 and i % 10 == 0:
                print(f"训练基学习器 {i+1}/{self.n_estimators}")

            # 1. 样本采样
            X_sample, y_sample, sample_indices = self._bootstrap_sample(X, y, n_subsample)

            # 2. 特征采样
            X_sample_selected, feature_indices = self._feature_sample(X_sample)
            self.estimator_features_.append(feature_indices)  # 保存特征索引

            # 3. 训练基学习器
            estimator = deepcopy(self.base_estimator)  # 深拷贝基学习器
            estimator.fit(X_sample_selected, y_sample)  # 训练基学习器
            self.estimators_.append(estimator)  # 添加到基学习器列表

            # 4. 更新OOB估计
            if self.oob_score:
                self._update_oob_estimation(i, X, sample_indices, feature_indices,
                                          oob_pred_sum, oob_count)

        # 最终处理
        self._finalize_fit(X, y, oob_pred_sum, oob_count)  # 完成训练
        return self  # 返回自身支持链式调用

    def predict(self, X):
        """预测类别 - 多数投票"""
        if not self.is_fitted:  # 检查模型是否已训练
            raise ValueError("请先调用 fit() 方法训练模型")

        # 获取所有基学习器的预测
        predictions = []  # 存储所有预测结果
        for estimator, feature_indices in zip(self.estimators_, self.estimator_features_):
            X_selected = X[:, feature_indices]  # 选择特征子集
            pred = estimator.predict(X_selected)  # 基学习器预测
            predictions.append(pred)  # 添加到预测列表

        # 多数投票
        predictions = np.array(predictions)  # 转换为数组 [n_estimators, n_samples]
        final_predictions = []  # 存储最终预测结果

        for sample_idx in range(predictions.shape[1]):  # 遍历每个样本
            # 统计当前样本在所有基学习器中的预测结果
            counts = np.bincount(predictions[:, sample_idx], minlength=self.n_classes_)
            final_predictions.append(np.argmax(counts))  # 选择票数最多的类别

        return np.array(final_predictions)  # 返回最终预测数组

    def predict_proba(self, X):
        """预测概率 - 平均概率"""
        if not self.is_fitted:  # 检查模型是否已训练
            raise ValueError("请先调用 fit() 方法训练模型")

        probas = []  # 存储所有基学习器的概率预测
        for estimator, feature_indices in zip(self.estimators_, self.estimator_features_):
            X_selected = X[:, feature_indices]  # 选择特征子集
            proba = estimator.predict_proba(X_selected)  # 基学习器概率预测
            probas.append(proba)  # 添加到概率列表

        # 平均所有基学习器的概率
        avg_proba = np.mean(probas, axis=0)  # 沿基学习器维度求平均
        return avg_proba  # 返回平均概率

    def _validate_data(self, X, y):
        """数据验证"""
        X = np.array(X)  # 转换为NumPy数组
        y = np.array(y)  # 转换为NumPy数组

        if len(X) != len(y):  # 检查样本数是否一致
            raise ValueError(f"X和y的长度不匹配: {len(X)} != {len(y)}")

        return X, y  # 返回转换后的数据

    def _bootstrap_sample(self, X, y, n_samples):
        """自助采样"""
        n_total = len(X)  # 总样本数

        if self.bootstrap:  # 有放回采样
            indices = np.random.choice(n_total, size=n_samples, replace=True)  # 有放回抽样
        else:  # 无放回采样
            indices = np.random.choice(n_total, size=n_samples, replace=False)  # 无放回抽样

        return X[indices], y[indices], indices  # 返回采样后的数据和索引

    def _feature_sample(self, X):
        """特征采样"""
        n_features = X.shape[1]  # 特征数
        n_selected = int(n_features * self.max_features)  # 计算选择的特征数

        if n_selected == 0:  # 检查是否有特征被选中
            raise ValueError("max_features太小，至少选择一个特征")

        if self.bootstrap_features:  # 有放回特征采样
            feature_indices = np.random.choice(n_features, size=n_selected, replace=True)  # 有放回
        else:  # 无放回特征采样
            feature_indices = np.random.choice(n_features, size=n_selected, replace=False)  # 无放回

        return X[:, feature_indices], feature_indices  # 返回特征子集和索引

    def _update_oob_estimation(self, estimator_idx, X, sample_indices,
                             feature_indices, oob_pred_sum, oob_count):
        """更新OOB估计"""
        n_samples = X.shape[0]  # 总样本数

        # 找出OOB样本（未被当前基学习器采样的样本）
        all_indices = set(range(n_samples))  # 所有样本索引
        in_bag_indices = set(sample_indices)  # 袋内样本索引
        oob_indices = list(all_indices - in_bag_indices)  # 袋外样本索引

        if len(oob_indices) == 0:  # 如果没有袋外样本
            return  # 直接返回

        # 用当前基学习器预测OOB样本
        estimator = self.estimators_[estimator_idx]  # 获取当前基学习器
        X_oob = X[oob_indices]  # 袋外样本特征
        X_oob_selected = X_oob[:, feature_indices]  # 选择特征子集

        # 检查基学习器是否有predict_proba方法
        if hasattr(estimator, 'predict_proba'):  # 如果有概率预测方法
            y_oob_pred = estimator.predict_proba(X_oob_selected)  # 获取概率预测
        else:  # 如果没有概率预测方法
            y_pred = estimator.predict(X_oob_selected)  # 获取类别预测
            y_oob_pred = np.zeros((len(y_pred), self.n_classes_))  # 创建概率矩阵
            for i, cls in enumerate(y_pred):  # 遍历预测类别
                y_oob_pred[i, cls] = 1.0  # 将对应类别设为1.0

        # 累加预测
        for idx, pred in zip(oob_indices, y_oob_pred):  # 遍历袋外样本
            oob_pred_sum[idx] += pred  # 累加概率预测
            oob_count[idx] += 1  # 增加预测计数

    def _finalize_fit(self, X, y, oob_pred_sum=None, oob_count=None):
        """完成训练"""
        self.is_fitted = True  # 标记模型已训练
        self.classes_ = np.unique(y)  # 获取所有类别
        self.n_classes_ = len(self.classes_)  # 获取类别数
        self.n_features_ = X.shape[1]  # 获取特征数

        # 计算OOB分数
        if self.oob_score and oob_pred_sum is not None and oob_count is not None:
            valid_oob = oob_count > 0  # 找出至少被一个基学习器预测过的样本
            if np.any(valid_oob):  # 如果有有效的OOB样本
                # 计算平均概率预测
                self.oob_decision_function_ = oob_pred_sum[valid_oob] / oob_count[valid_oob, np.newaxis]
                y_oob_pred = np.argmax(self.oob_decision_function_, axis=1)  # 获取预测类别
                y_oob_true = y[valid_oob]  # 获取真实标签
                self.oob_score_ = accuracy_score(y_oob_true, y_oob_pred)  # 计算准确率

                if self.verbose > 0:  # 如果设置了详细输出
                    print(f"OOB分数: {self.oob_score_:.4f} (基于{np.sum(valid_oob)}个样本)")
            else:  # 如果没有有效的OOB样本
                warnings.warn("没有有效的OOB样本，无法计算OOB分数")  # 发出警告
                self.oob_score_ = None  # 设置OOB分数为None
        else:  # 如果不计算OOB分数
            self.oob_score_ = None  # 设置OOB分数为None

    def get_feature_importances(self):
        """获取特征重要性（Bagging没有天然的特征重要性，返回None）"""
        warnings.warn("BaggingClassifier没有内置的特征重要性计算方法")  # 发出警告
        return None  # 返回None
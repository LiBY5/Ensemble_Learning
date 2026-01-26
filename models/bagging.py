"""Bagging与随机森林实现 - models/bagging.py (修复版)"""
import numpy as np
from copy import deepcopy
from typing import List, Optional, Union, Tuple
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import resample
from sklearn.metrics import accuracy_score
import warnings

from models.base import BaseClassifier


class BaggingClassifier(BaseClassifier):
    """Bagging分类器"""

    def __init__(self,
                 base_estimator: BaseEstimator,
                 n_estimators: int = 10,
                 max_samples: float = 1.0,
                 max_features: float = 1.0,
                 bootstrap: bool = True,
                 bootstrap_features: bool = False,
                 oob_score: bool = False,
                 random_state: Optional[int] = None,
                 verbose: int = 0):
        """
        参数初始化
        """
        super().__init__(name="BaggingClassifier")
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
        self.estimators_ = []  # 存储所有基学习器
        self.estimator_features_ = []  # 存储每个基学习器使用的特征索引
        self.oob_decision_function_ = None
        self.oob_score_ = None

        # 设置随机种子
        if random_state is not None:
            np.random.seed(random_state)

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
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def fit(self, X, y):
        """训练Bagging分类器"""
        # 输入验证
        X, y = self._validate_data(X, y)

        n_samples, n_features = X.shape
        n_subsample = int(n_samples * self.max_samples)

        # 初始化OOB数据结构
        oob_pred_sum = None
        oob_count = None

        if self.oob_score:
            n_classes = len(np.unique(y))
            oob_pred_sum = np.zeros((n_samples, n_classes))
            oob_count = np.zeros(n_samples)

        # 清空之前的结果
        self.estimators_ = []
        self.estimator_features_ = []

        # 训练n_estimators个基学习器
        for i in range(self.n_estimators):
            if self.verbose > 0 and i % 10 == 0:
                print(f"训练基学习器 {i+1}/{self.n_estimators}")

            # 1. 样本采样
            X_sample, y_sample, sample_indices = self._bootstrap_sample(X, y, n_subsample)

            # 2. 特征采样
            X_sample_selected, feature_indices = self._feature_sample(X_sample)
            self.estimator_features_.append(feature_indices)

            # 3. 训练基学习器
            estimator = deepcopy(self.base_estimator)
            estimator.fit(X_sample_selected, y_sample)
            self.estimators_.append(estimator)

            # 4. 更新OOB估计
            if self.oob_score:
                self._update_oob_estimation(i, X, sample_indices, feature_indices,
                                          oob_pred_sum, oob_count)

        # 最终处理
        self._finalize_fit(X, y, oob_pred_sum, oob_count)
        return self

    def predict(self, X):
        """预测类别 - 多数投票"""
        if not self.is_fitted:
            raise ValueError("请先调用 fit() 方法训练模型")

        # 获取所有基学习器的预测
        predictions = []
        for estimator, feature_indices in zip(self.estimators_, self.estimator_features_):
            X_selected = X[:, feature_indices]
            pred = estimator.predict(X_selected)
            predictions.append(pred)

        # 多数投票
        predictions = np.array(predictions)  # [n_estimators, n_samples]
        final_predictions = []

        for sample_idx in range(predictions.shape[1]):
            counts = np.bincount(predictions[:, sample_idx], minlength=self.n_classes_)
            final_predictions.append(np.argmax(counts))

        return np.array(final_predictions)

    def predict_proba(self, X):
        """预测概率 - 平均概率"""
        if not self.is_fitted:
            raise ValueError("请先调用 fit() 方法训练模型")

        probas = []
        for estimator, feature_indices in zip(self.estimators_, self.estimator_features_):
            X_selected = X[:, feature_indices]
            proba = estimator.predict_proba(X_selected)
            probas.append(proba)

        # 平均所有基学习器的概率
        avg_proba = np.mean(probas, axis=0)
        return avg_proba

    def _validate_data(self, X, y):
        """数据验证"""
        X = np.array(X)
        y = np.array(y)

        if len(X) != len(y):
            raise ValueError(f"X和y的长度不匹配: {len(X)} != {len(y)}")

        return X, y

    def _bootstrap_sample(self, X, y, n_samples):
        """自助采样"""
        n_total = len(X)

        if self.bootstrap:
            indices = np.random.choice(n_total, size=n_samples, replace=True)
        else:
            # 无放回采样
            indices = np.random.choice(n_total, size=n_samples, replace=False)

        return X[indices], y[indices], indices

    def _feature_sample(self, X):
        """特征采样"""
        n_features = X.shape[1]
        n_selected = int(n_features * self.max_features)

        if n_selected == 0:
            raise ValueError("max_features太小，至少选择一个特征")

        if self.bootstrap_features:
            # 有放回特征采样
            feature_indices = np.random.choice(n_features, size=n_selected, replace=True)
        else:
            # 无放回特征采样
            feature_indices = np.random.choice(n_features, size=n_selected, replace=False)

        return X[:, feature_indices], feature_indices

    def _update_oob_estimation(self, estimator_idx, X, sample_indices,
                             feature_indices, oob_pred_sum, oob_count):
        """更新OOB估计"""
        n_samples = X.shape[0]

        # 找出OOB样本
        all_indices = set(range(n_samples))
        in_bag_indices = set(sample_indices)
        oob_indices = list(all_indices - in_bag_indices)

        if len(oob_indices) == 0:
            return

        # 用当前基学习器预测OOB样本
        estimator = self.estimators_[estimator_idx]
        X_oob = X[oob_indices]
        X_oob_selected = X_oob[:, feature_indices]

        # 检查基学习器是否有predict_proba方法
        if hasattr(estimator, 'predict_proba'):
            y_oob_pred = estimator.predict_proba(X_oob_selected)
        else:
            # 如果没有predict_proba，使用predict并转换为one-hot
            y_pred = estimator.predict(X_oob_selected)
            y_oob_pred = np.zeros((len(y_pred), self.n_classes_))
            for i, cls in enumerate(y_pred):
                y_oob_pred[i, cls] = 1.0

        # 累加预测
        for idx, pred in zip(oob_indices, y_oob_pred):
            oob_pred_sum[idx] += pred
            oob_count[idx] += 1

    def _finalize_fit(self, X, y, oob_pred_sum=None, oob_count=None):
        """完成训练"""
        self.is_fitted = True
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        self.n_features_ = X.shape[1]

        # 计算OOB分数
        if self.oob_score and oob_pred_sum is not None and oob_count is not None:
            valid_oob = oob_count > 0
            if np.any(valid_oob):
                self.oob_decision_function_ = oob_pred_sum[valid_oob] / oob_count[valid_oob, np.newaxis]
                y_oob_pred = np.argmax(self.oob_decision_function_, axis=1)
                y_oob_true = y[valid_oob]
                self.oob_score_ = accuracy_score(y_oob_true, y_oob_pred)

                if self.verbose > 0:
                    print(f"OOB分数: {self.oob_score_:.4f} (基于{np.sum(valid_oob)}个样本)")
            else:
                warnings.warn("没有有效的OOB样本，无法计算OOB分数")
                self.oob_score_ = None
        else:
            self.oob_score_ = None

    def get_feature_importances(self):
        """获取特征重要性（Bagging没有天然的特征重要性，返回None）"""
        warnings.warn("BaggingClassifier没有内置的特征重要性计算方法")
        return None
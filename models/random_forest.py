"""随机森林实现 - models/random_forest.py (修复版)"""
import numpy as np
from copy import deepcopy
from typing import Optional, Union
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import warnings

from models.base import BaseClassifier


class RandomForestClassifier(BaseClassifier):
    """随机森林分类器"""

    def __init__(self,
                 n_estimators: int = 100,
                 criterion: str = 'gini',
                 max_depth: Optional[int] = None,
                 min_samples_split: Union[int, float] = 2,
                 min_samples_leaf: Union[int, float] = 1,
                 max_features: Union[str, float] = 'sqrt',
                 bootstrap: bool = True,
                 oob_score: bool = False,
                 random_state: Optional[int] = None,
                 n_jobs: Optional[int] = None,
                 verbose: int = 0):
        super().__init__(name="RandomForestClassifier")

        # 树的数量和配置
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features

        # Bagging相关参数
        self.bootstrap = bootstrap
        self.oob_score = oob_score

        # 并行化和随机性
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose

        # 内部状态
        self.estimators_ = []
        self.estimator_features_ = []  # 记录每棵树使用的特征索引
        self.feature_importances_ = None
        self.oob_score_ = None

        # 设置随机种子
        if random_state is not None:
            np.random.seed(random_state)

    def get_params(self, deep=True):
        """获取模型参数 - sklearn兼容性"""
        return {
            'n_estimators': self.n_estimators,
            'criterion': self.criterion,
            'max_depth': self.max_depth,
            'min_samples_split': self.min_samples_split,
            'min_samples_leaf': self.min_samples_leaf,
            'max_features': self.max_features,
            'bootstrap': self.bootstrap,
            'oob_score': self.oob_score,
            'random_state': self.random_state,
            'n_jobs': self.n_jobs,
            'verbose': self.verbose
        }

    def set_params(self, **params):
        """设置模型参数 - sklearn兼容性"""
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def fit(self, X, y):
        """训练随机森林"""
        # 数据验证
        X, y = self._validate_data(X, y)
        n_samples, n_features = X.shape

        # 计算实际的最大特征数
        max_features = self._compute_max_features(n_features)

        # 初始化OOB估计
        oob_pred_sum = None
        oob_count = None

        if self.oob_score:
            n_classes = len(np.unique(y))
            oob_pred_sum = np.zeros((n_samples, n_classes))
            oob_count = np.zeros(n_samples)

        # 清空之前的模型
        self.estimators_ = []
        self.estimator_features_ = []

        # 训练每棵树
        for i in range(self.n_estimators):
            if self.verbose > 0 and i % 10 == 0:
                print(f"训练决策树 {i+1}/{self.n_estimators}")

            # 1. 样本采样
            if self.bootstrap:
                indices = np.random.choice(n_samples, size=n_samples, replace=True)
                X_sample = X[indices]
                y_sample = y[indices]
                sample_indices = indices
            else:
                X_sample, y_sample = X, y
                sample_indices = np.arange(n_samples)

            # 2. 特征采样（随机森林的关键）
            feature_indices = np.random.choice(
                n_features,
                size=max_features,
                replace=False
            )
            self.estimator_features_.append(feature_indices)

            # 3. 训练决策树
            tree = DecisionTreeClassifier(
                criterion=self.criterion,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                random_state=self.random_state + i if self.random_state is not None else None
            )

            # 使用特征子集训练
            tree.fit(X_sample[:, feature_indices], y_sample)
            self.estimators_.append(tree)

            # 4. 更新OOB估计
            if self.oob_score and self.bootstrap:
                self._update_oob_estimation(
                    i, X, y, sample_indices, feature_indices,
                    oob_pred_sum, oob_count
                )

        # 5. 计算特征重要性
        self.feature_importances_ = self._compute_feature_importances(n_features)

        # 6. 完成训练
        self._finalize_fit(X, y, oob_pred_sum, oob_count)

        return self

    def predict(self, X):
        """预测类别 - 多数投票"""
        if not self.is_fitted:
            raise ValueError("模型必须先训练")

        # 收集所有树的预测
        all_predictions = []

        for tree, feature_indices in zip(self.estimators_, self.estimator_features_):
            # 使用特征子集进行预测
            X_selected = X[:, feature_indices]
            pred = tree.predict(X_selected)
            all_predictions.append(pred)

        # 转为数组：[n_trees, n_samples]
        all_predictions = np.array(all_predictions)

        # 多数投票
        final_predictions = []
        for sample_idx in range(all_predictions.shape[1]):
            counts = np.bincount(all_predictions[:, sample_idx], minlength=self.n_classes_)
            final_predictions.append(np.argmax(counts))

        return np.array(final_predictions)

    def predict_proba(self, X):
        """预测概率 - 平均概率"""
        if not self.is_fitted:
            raise ValueError("模型必须先训练")

        # 收集所有树的概率预测
        all_probas = []

        for tree, feature_indices in zip(self.estimators_, self.estimator_features_):
            X_selected = X[:, feature_indices]
            proba = tree.predict_proba(X_selected)
            all_probas.append(proba)

        # 平均所有树的概率
        avg_proba = np.mean(all_probas, axis=0)
        return avg_proba

    def _validate_data(self, X, y):
        """数据验证"""
        X = np.array(X)
        y = np.array(y)

        if len(X) != len(y):
            raise ValueError(f"X和y的长度不匹配: {len(X)} != {len(y)}")

        return X, y

    def _compute_max_features(self, n_features):
        """计算每次分裂考虑的特征数"""
        if isinstance(self.max_features, str):
            if self.max_features == 'auto':
                return int(np.sqrt(n_features))
            elif self.max_features == 'sqrt':
                return int(np.sqrt(n_features))
            elif self.max_features == 'log2':
                return int(np.log2(n_features))
            else:
                raise ValueError(f"不支持的max_features字符串: {self.max_features}")
        elif isinstance(self.max_features, float):
            # 比例
            if self.max_features <= 0.0 or self.max_features > 1.0:
                raise ValueError("max_features比例必须在(0, 1]范围内")
            return max(1, int(self.max_features * n_features))
        elif isinstance(self.max_features, int):
            if self.max_features <= 0:
                raise ValueError("max_features必须为正整数")
            return min(self.max_features, n_features)
        else:
            raise TypeError("max_features必须是字符串、浮点数或整数")

    def _update_oob_estimation(self, tree_idx, X, y, sample_indices,
                             feature_indices, oob_pred_sum, oob_count):
        """更新OOB估计"""
        n_samples = X.shape[0]

        # 找出OOB样本
        all_indices = set(range(n_samples))
        in_bag_indices = set(sample_indices)
        oob_indices = list(all_indices - in_bag_indices)

        if len(oob_indices) == 0:
            return

        # 用当前树预测OOB样本
        tree = self.estimators_[tree_idx]
        X_oob = X[oob_indices]
        X_oob_selected = X_oob[:, feature_indices]
        y_oob_pred_proba = tree.predict_proba(X_oob_selected)

        # 累加预测
        for idx, pred in zip(oob_indices, y_oob_pred_proba):
            oob_pred_sum[idx] += pred
            oob_count[idx] += 1

    def _compute_feature_importances(self, n_features):
        """计算基尼重要性"""
        importances = np.zeros(n_features)

        for tree, feature_indices in zip(self.estimators_, self.estimator_features_):
            # 获取树的特征重要性
            tree_importances = tree.feature_importances_

            # 将重要性值加到对应的原始特征位置上
            for idx, importance in zip(feature_indices, tree_importances):
                importances[idx] += importance

        # 归一化
        if importances.sum() > 0:
            importances /= importances.sum()

        return importances

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
                oob_decision = oob_pred_sum[valid_oob] / oob_count[valid_oob, np.newaxis]
                y_oob_pred = np.argmax(oob_decision, axis=1)
                y_oob_true = y[valid_oob]
                self.oob_score_ = accuracy_score(y_oob_true, y_oob_pred)

                if self.verbose > 0:
                    print(f"OOB分数: {self.oob_score_:.4f} (基于{np.sum(valid_oob)}个样本)")
            else:
                warnings.warn("没有有效的OOB样本，无法计算OOB分数")
                self.oob_score_ = None
        else:
            self.oob_score_ = None
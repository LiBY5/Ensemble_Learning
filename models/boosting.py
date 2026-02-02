"""Boosting算法实现"""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from copy import deepcopy


class AdaBoostClassifier(BaseEstimator, ClassifierMixin):
    """AdaBoost分类器"""

    def __init__(self,
                 base_estimator=None,
                 n_estimators=50,
                 learning_rate=1.0,
                 algorithm='SAMME',
                 random_state=None):
        """
        参数说明:
        ----------
        base_estimator : 基学习器，默认为决策树桩（深度为1的决策树）
        n_estimators : 基学习器数量
        learning_rate : 学习率，控制每个基学习器的贡献
        algorithm : 'SAMME' 或 'SAMME.R'，多分类算法
        random_state : 随机种子
        """
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.algorithm = algorithm
        self.random_state = random_state

        # 设置默认基学习器
        if self.base_estimator is None:
            self.base_estimator = DecisionTreeClassifier(max_depth=1)

        # 设置随机种子
        if random_state is not None:
            np.random.seed(random_state)

        # 初始化存储结构
        self.estimators_ = []  # 存储所有基学习器
        self.estimator_weights_ = np.zeros(self.n_estimators)  # 基学习器权重
        self.estimator_errors_ = np.zeros(self.n_estimators)  # 每个基学习器的错误率
        self.classes_ = None  # 存储类别标签

    def fit(self, X, y, sample_weight=None):
        """训练AdaBoost模型"""
        # 检查输入数据的有效性
        X, y = check_X_y(X, y)
        n_samples, n_features = X.shape

        # 获取所有类别
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)

        # 将标签转换为适合AdaBoost的形式
        if n_classes == 2:
            # 二分类：将标签转换为{-1, 1}
            y_coded = np.where(y == self.classes_[0], -1, 1)
            self._binary = True
        else:
            # 多分类：使用one-hot编码
            y_coded = np.eye(n_classes)[y]
            self._binary = False

        # 初始化样本权重
        if sample_weight is None:
            sample_weight = np.ones(n_samples) / n_samples
        else:
            sample_weight = np.array(sample_weight) / np.sum(sample_weight)

        # 清空之前的训练结果
        self.estimators_ = []
        self.estimator_weights_ = np.zeros(self.n_estimators)
        self.estimator_errors_ = np.zeros(self.n_estimators)

        # 主训练循环
        for t in range(self.n_estimators):
            print(f"\n第 {t + 1}/{self.n_estimators} 轮训练...")

            # 1. 使用当前权重训练基学习器
            estimator = self._make_estimator()

            if self._binary:
                estimator.fit(X, y_coded, sample_weight=sample_weight)
                y_pred = estimator.predict(X)
            else:
                estimator.fit(X, y, sample_weight=sample_weight)
                y_pred = estimator.predict(X)
                # 转换为one-hot形式
                y_pred_coded = np.eye(n_classes)[np.searchsorted(self.classes_, y_pred)]

            # 2. 计算加权错误率
            if self._binary:
                incorrect = (y_pred != y_coded)
                estimator_error = np.dot(sample_weight, incorrect)
            else:
                if self.algorithm == 'SAMME':
                    incorrect = (y_pred != y)
                    estimator_error = np.dot(sample_weight, incorrect)
                else:  # SAMME.R
                    # 使用概率估计
                    y_proba = estimator.predict_proba(X)
                    y_proba = np.clip(y_proba, 1e-15, 1 - 1e-15)

                    # 计算加权对数概率
                    log_proba = np.log(y_proba)
                    inner_product = np.sum(y_coded * log_proba, axis=1)
                    estimator_error = np.dot(sample_weight, 1 - inner_product)

            print(f"  加权错误率: {estimator_error:.4f}")

            # 如果错误率≥0.5，提前停止
            if estimator_error >= 0.5:
                print(f"  错误率≥0.5，提前停止训练")
                self.n_estimators = t
                self.estimator_weights_ = self.estimator_weights_[:t]
                self.estimator_errors_ = self.estimator_errors_[:t]
                break

            # 如果错误率为0，也提前停止
            if estimator_error <= 0:
                print(f"  错误率为0，完美分类，提前停止")
                self.estimator_weights_[t] = 1.0
                self.estimator_errors_[t] = 0
                self.estimators_.append(estimator)
                self.n_estimators = t + 1
                break

            # 3. 计算基学习器权重
            if self.algorithm == 'SAMME' or self._binary:
                if self._binary:
                    # 二分类
                    estimator_weight = self.learning_rate * 0.5 * np.log(
                        (1 - estimator_error) / estimator_error
                    )
                else:
                    # 多分类SAMME
                    estimator_weight = self.learning_rate * (
                            0.5 * np.log((1 - estimator_error) / estimator_error) +
                            np.log(n_classes - 1)
                    )
            else:  # SAMME.R
                # SAMME.R使用不同的权重计算方式
                estimator_weight = 1.0

            self.estimator_weights_[t] = estimator_weight
            self.estimator_errors_[t] = estimator_error
            self.estimators_.append(estimator)

            print(f"  基学习器权重: {estimator_weight:.4f}")

            # 4. 更新样本权重
            if self.algorithm == 'SAMME' or self._binary:
                if self._binary:
                    # 二分类更新公式
                    sample_weight *= np.exp(
                        -estimator_weight * y_coded * y_pred
                    )
                else:
                    # 多分类SAMME更新
                    incorrect = (y_pred != y)
                    sample_weight *= np.exp(
                        estimator_weight * incorrect
                    )
            else:  # SAMME.R
                # 使用概率更新权重
                sample_weight *= np.exp(
                    -((n_classes - 1) / n_classes) *
                    np.sum(y_coded * np.log(y_proba), axis=1)
                )

            # 归一化权重
            sample_weight /= np.sum(sample_weight)

            # 打印权重信息
            print(f"  样本权重范围: [{sample_weight.min():.6f}, {sample_weight.max():.6f}]")
            print(f"  平均样本权重: {sample_weight.mean():.6f}")

        return self

    def _make_estimator(self):
        """创建基学习器的深拷贝"""
        return deepcopy(self.base_estimator)

    def predict(self, X):
        """预测类别"""
        check_is_fitted(self)
        X = check_array(X)

        if self._binary:
            # 二分类：使用决策函数
            pred = self.decision_function(X)
            return np.where(pred > 0, self.classes_[1], self.classes_[0])
        else:
            # 多分类
            n_samples = X.shape[0]
            n_classes = len(self.classes_)

            pred = np.zeros((n_samples, n_classes))

            for estimator, weight in zip(self.estimators_, self.estimator_weights_):
                if self.algorithm == 'SAMME':
                    y_pred = estimator.predict(X)
                    pred[np.arange(n_samples),
                    np.searchsorted(self.classes_, y_pred)] += weight
                else:  # SAMME.R
                    y_proba = estimator.predict_proba(X)
                    pred += weight * y_proba

            return self.classes_[np.argmax(pred, axis=1)]

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
    """AdaBoost回归器"""

    def __init__(self, base_estimator=None, n_estimators=50,
                 learning_rate=1.0, loss='linear', random_state=None):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.loss = loss  # 'linear', 'square', 'exponential'
        self.random_state = random_state

        if self.base_estimator is None:
            from sklearn.tree import DecisionTreeRegressor
            self.base_estimator = DecisionTreeRegressor(max_depth=3)

        if random_state is not None:
            np.random.seed(random_state)

    def fit(self, X, y, sample_weight=None):
        """训练AdaBoost回归器"""
        X, y = check_X_y(X, y)
        n_samples = X.shape[0]

        # 初始化
        if sample_weight is None:
            sample_weight = np.ones(n_samples) / n_samples

        self.estimators_ = []
        self.estimator_weights_ = np.zeros(self.n_estimators)
        self.estimator_errors_ = np.zeros(self.n_estimators)

        # 初始预测
        y_pred = np.zeros(n_samples)

        for t in range(self.n_estimators):
            print(f"\n第 {t + 1}/{self.n_estimators} 轮回归训练...")

            # 1. 训练基学习器
            estimator = deepcopy(self.base_estimator)
            estimator.fit(X, y, sample_weight=sample_weight)
            y_pred_i = estimator.predict(X)

            # 2. 计算损失
            error_vector = y - (y_pred + y_pred_i)

            if self.loss == 'linear':
                loss_vector = np.abs(error_vector)
            elif self.loss == 'square':
                loss_vector = error_vector ** 2
            elif self.loss == 'exponential':
                loss_vector = 1 - np.exp(-np.abs(error_vector))

            # 3. 计算加权平均损失
            estimator_error = np.dot(sample_weight, loss_vector)
            self.estimator_errors_[t] = estimator_error
            print(f"  加权损失: {estimator_error:.4f}")

            # 4. 计算基学习器权重
            if estimator_error >= 1.0:
                print(f"  损失≥1.0，提前停止")
                break

            estimator_weight = self.learning_rate * np.log(
                (1 - estimator_error) / estimator_error
            )
            self.estimator_weights_[t] = estimator_weight
            self.estimators_.append(estimator)

            print(f"  基学习器权重: {estimator_weight:.4f}")

            # 5. 更新样本权重
            sample_weight *= np.exp(estimator_weight * loss_vector)
            sample_weight /= sample_weight.sum()

            # 6. 更新累计预测
            y_pred += estimator_weight * y_pred_i

        return self

    def predict(self, X):
        """预测"""
        check_is_fitted(self)
        X = check_array(X)

        y_pred = np.zeros(X.shape[0])
        for estimator, weight in zip(self.estimators_, self.estimator_weights_):
            y_pred += weight * estimator.predict(X)

        return y_pred
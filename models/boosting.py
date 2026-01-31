"""Boosting算法实现"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
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
        参数:
        ----------
        base_estimator : 基学习器（默认决策树桩）
        n_estimators : 基学习器数量
        learning_rate : 学习率，收缩每个基学习器的贡献
        algorithm : 'SAMME' 或 'SAMME.R'
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

        if random_state is not None:
            np.random.seed(random_state)

        # 存储训练结果的属性
        self.estimators_ = []
        self.estimator_weights_ = np.zeros(self.n_estimators)
        self.estimator_errors_ = np.zeros(self.n_estimators)
        self.classes_ = None

    def _check_params(self):
        """检查参数"""
        if self.learning_rate <= 0:
            raise ValueError("learning_rate必须大于0")
        if self.n_estimators <= 0:
            raise ValueError("n_estimators必须大于0")

    def fit(self, X, y, sample_weight=None):
        """训练AdaBoost模型"""
        # 检查输入
        X, y = check_X_y(X, y)
        n_samples, n_features = X.shape

        # 初始化
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)

        # 将标签转换为{-1, 1}或one-hot编码
        if n_classes == 2:
            # 二分类：转换为{-1, 1}
            y_coded = np.where(y == self.classes_[0], -1, 1)
            self._binary = True
        else:
            # 多分类：使用SAMME算法
            y_coded = np.eye(n_classes)[y]
            self._binary = False

        # 初始化样本权重
        if sample_weight is None:
            sample_weight = np.ones(n_samples) / n_samples
        else:
            sample_weight = np.array(sample_weight) / np.sum(sample_weight)

        self.estimators_ = []
        self.estimator_weights_ = np.zeros(self.n_estimators)
        self.estimator_errors_ = np.zeros(self.n_estimators)

        print(f"开始训练AdaBoost，共{self.n_estimators}轮...")

        for t in range(self.n_estimators):
            # 1. 使用当前权重训练基学习器
            estimator = self._make_estimator()

            if self._binary:
                estimator.fit(X, y_coded, sample_weight=sample_weight)
                y_pred = estimator.predict(X)
            else:
                # 多分类
                estimator.fit(X, y, sample_weight=sample_weight)
                y_pred = estimator.predict(X)
                # 转换为类别索引
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
                    y_proba = np.clip(y_proba, 1e-15, 1 - 1e-15)  # 避免log(0)

                    # 计算加权对数概率
                    log_proba = np.log(y_proba)
                    inner_product = np.sum(y_coded * log_proba, axis=1)
                    estimator_error = np.dot(sample_weight, 1 - inner_product)

            # 如果错误率为0或≥0.5，提前停止
            if estimator_error >= 0.5:
                print(f"第{t+1}轮错误率≥0.5 ({estimator_error:.4f})，提前停止")
                self.n_estimators = t
                self.estimator_weights_ = self.estimator_weights_[:t]
                self.estimator_errors_ = self.estimator_errors_[:t]
                break

            if estimator_error <= 0:
                print(f"第{t+1}轮错误率为0，提前停止")
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
                        np.log((1 - estimator_error) / estimator_error) +
                        np.log(n_classes - 1)
                    )
            else:  # SAMME.R
                # SAMME.R使用不同的权重计算方式
                estimator_weight = 1.0

            self.estimator_weights_[t] = estimator_weight
            self.estimator_errors_[t] = estimator_error
            self.estimators_.append(estimator)

            # 4. 更新样本权重
            if self.algorithm == 'SAMME' or self._binary:
                if self._binary:
                    # 二分类更新
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

            # 打印进度
            if (t + 1) % 10 == 0 or t == 0 or t == self.n_estimators - 1:
                print(f"  第{t+1:3d}轮: 错误率={estimator_error:.4f}, 权重={estimator_weight:.4f}")

        print(f"训练完成，实际使用了{self.n_estimators}个基学习器")
        return self

    def _make_estimator(self):
        """创建基学习器实例"""
        return deepcopy(self.base_estimator)

    def predict(self, X):
        """预测类别"""
        check_is_fitted(self)
        X = check_array(X)

        if self._binary:
            # 二分类
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


class AdaBoostRegressor(BaseEstimator):
    """AdaBoost回归器"""

    def __init__(self, base_estimator=None, n_estimators=50,
                 learning_rate=1.0, loss='square', random_state=None):
        """
        参数:
        ----------
        base_estimator : 基学习器（默认决策树）
        n_estimators : 基学习器数量
        learning_rate : 学习率
        loss : 损失函数类型，'square'、'linear' 或 'exponential'
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

        self.estimators_ = []
        self.estimator_weights_ = np.zeros(self.n_estimators)
        self.estimator_errors_ = np.zeros(self.n_estimators)

    def _make_estimator(self):
        """创建基学习器实例"""
        return deepcopy(self.base_estimator)

    def _compute_loss(self, y_true, y_pred):
        """计算损失向量"""
        error = y_true - y_pred

        if self.loss == 'square':
            return error ** 2
        elif self.loss == 'linear':
            return np.abs(error)
        elif self.loss == 'exponential':
            return 1 - np.exp(-np.abs(error))
        else:
            raise ValueError(f"未知损失函数: {self.loss}")

    def fit(self, X, y, sample_weight=None):
        """训练AdaBoost回归器"""
        X, y = check_X_y(X, y)
        n_samples = X.shape[0]

        # 初始化样本权重
        if sample_weight is None:
            sample_weight = np.ones(n_samples) / n_samples
        else:
            sample_weight = np.array(sample_weight) / np.sum(sample_weight)

        # 存储模型和权重
        self.estimators_ = []
        self.estimator_weights_ = np.zeros(self.n_estimators)
        self.estimator_errors_ = np.zeros(self.n_estimators)

        # 初始化预测
        y_pred = np.zeros(n_samples)

        print(f"开始训练AdaBoost回归器，共{self.n_estimators}轮...")

        for t in range(self.n_estimators):
            # 1. 计算残差
            residual = y - y_pred

            # 2. 训练基学习器拟合残差
            estimator = self._make_estimator()
            estimator.fit(X, residual, sample_weight=sample_weight)
            y_pred_i = estimator.predict(X)

            # 3. 计算加权损失
            loss_vector = self._compute_loss(y, y_pred + y_pred_i)
            estimator_error = np.dot(sample_weight, loss_vector)

            # 如果误差过大，提前停止
            if estimator_error >= 1.0 or estimator_error <= 0:
                print(f"第{t+1}轮误差异常 ({estimator_error:.4f})，提前停止")
                self.n_estimators = t
                self.estimator_weights_ = self.estimator_weights_[:t]
                self.estimator_errors_ = self.estimator_errors_[:t]
                break

            # 4. 计算基学习器权重
            estimator_weight = self.learning_rate * np.log(
                (1 - estimator_error) / estimator_error
            ) if estimator_error > 0 else 1.0

            self.estimator_weights_[t] = estimator_weight
            self.estimator_errors_[t] = estimator_error
            self.estimators_.append(estimator)

            # 5. 更新样本权重
            sample_weight *= np.exp(estimator_weight * loss_vector)
            sample_weight /= sample_weight.sum()  # 归一化

            # 6. 更新累计预测
            y_pred += estimator_weight * y_pred_i

            # 打印进度
            if (t + 1) % 10 == 0 or t == 0 or t == self.n_estimators - 1:
                print(f"  第{t+1:3d}轮: 误差={estimator_error:.4f}, 权重={estimator_weight:.4f}")

        print(f"回归器训练完成，实际使用了{self.n_estimators}个基学习器")
        return self

    def predict(self, X):
        """预测"""
        check_is_fitted(self)
        X = check_array(X)

        y_pred = np.zeros(X.shape[0])
        for estimator, weight in zip(self.estimators_, self.estimator_weights_):
            y_pred += weight * estimator.predict(X)

        return y_pred

    def score(self, X, y):
        """计算R²分数"""
        from sklearn.metrics import r2_score
        y_pred = self.predict(X)
        return r2_score(y, y_pred)
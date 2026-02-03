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

"实现梯度提升回归树（GBRT）"


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
        """返回初始估计器（通常为常数）"""
        raise NotImplementedError


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

class GradientBoostingRegressor(BaseEstimator):
    """梯度提升回归树"""

    def __init__(self,
                 loss='ls',  # 'ls', 'lad', 'huber', 'quantile'
                 learning_rate=0.1,
                 n_estimators=100,
                 subsample=1.0,
                 criterion='friedman_mse',
                 min_samples_split=2,
                 min_samples_leaf=1,
                 max_depth=3,
                 min_impurity_decrease=0.0,
                 init=None,
                 random_state=None,
                 max_features=None,
                 alpha=0.9,  # 用于huber和quantile损失
                 verbose=0,
                 max_leaf_nodes=None):

        self.loss = loss
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.subsample = subsample
        self.criterion = criterion
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth
        self.min_impurity_decrease = min_impurity_decrease
        self.init = init
        self.random_state = random_state
        self.max_features = max_features
        self.alpha = alpha
        self.verbose = verbose
        self.max_leaf_nodes = max_leaf_nodes

        if random_state is not None:
            np.random.seed(random_state)

        self.estimators_ = []
        self.train_score_ = []
        self.init_ = None
        self.loss_ = None

    def _init_state(self):
        """初始化状态"""
        self.estimators_ = []
        self.train_score_ = []

        # 初始化损失函数
        if self.loss == 'ls':
            self.loss_ = LeastSquaresError()
        elif self.loss == 'lad':
            self.loss_ = LeastAbsoluteError()
        elif self.loss == 'huber':
            self.loss_ = HuberLoss(self.alpha)
        elif self.loss == 'quantile':
            self.loss_ = QuantileLoss(self.alpha)
        else:
            raise ValueError(f"未知损失函数: {self.loss}")

    def _init_constant(self, y):
        """用常数初始化预测"""
        self.init_ = self.loss_.init_estimator()
        self.init_.fit(y)
        return self.init_.predict(np.zeros(len(y)))

    def fit(self, X, y, sample_weight=None):
        """训练梯度提升模型"""
        X, y = check_X_y(X, y)
        n_samples, n_features = X.shape

        # 初始化
        self._init_state()

        # 初始预测
        y_pred = self._init_constant(y)

        # 主循环
        for t in range(self.n_estimators):
            # 1. 计算负梯度（伪残差）
            negative_gradient = self.loss_.negative_gradient(y, y_pred)

            # 2. 子采样
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

            # 3. 训练决策树拟合负梯度
            tree = DecisionTreeRegressor(
                criterion=self.criterion,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                random_state=self.random_state,
                max_leaf_nodes=self.max_leaf_nodes
            )

            tree.fit(X_subset, y_subset, sample_weight=sample_weight_subset)
            self.estimators_.append(tree)

            # 4. 更新预测
            update = tree.predict(X)
            y_pred += self.learning_rate * update

            # 5. 记录训练分数
            self.train_score_.append(self.loss_(y, y_pred))

            if self.verbose > 0 and t % 10 == 0:
                print(f"Iteration {t}, loss = {self.train_score_[-1]:.4f}")

        return self

    def predict(self, X):
        """预测"""
        check_is_fitted(self)
        X = check_array(X)

        # 初始预测
        y_pred = self.init_.predict(np.zeros(X.shape[0]))

        # 累加树预测
        for tree in self.estimators_:
            y_pred += self.learning_rate * tree.predict(X)

        return y_pred

    def staged_predict(self, X):
        """按阶段预测"""
        check_is_fitted(self)
        X = check_array(X)

        # 初始预测
        y_pred = self.init_.predict(np.zeros(X.shape[0]))

        yield y_pred.copy()

        # 逐步添加树
        for tree in self.estimators_:
            y_pred += self.learning_rate * tree.predict(X)
            yield y_pred.copy()
class LossFunction:
    """损失函数基类"""

    def __init__(self):
        pass

    def __call__(self, y, pred):
        """计算损失值"""
        raise NotImplementedError

    def negative_gradient(self, y, pred):
        """计算负梯度"""
        raise NotImplementedError

    def init_estimator(self):
        """返回初始估计器"""
        raise NotImplementedError


class LeastSquaresError(LossFunction):
    """平方损失"""

    def __call__(self, y, pred):
        return np.mean((y - pred) ** 2)

    def negative_gradient(self, y, pred):
        return y - pred

    def init_estimator(self):
        class MeanEstimator:
            def fit(self, y):
                self.mean = np.mean(y)

            def predict(self, X):
                return np.full(X.shape[0], self.mean)

        return MeanEstimator()


class LeastAbsoluteError(LossFunction):
    """绝对损失"""

    def __call__(self, y, pred):
        return np.mean(np.abs(y - pred))

    def negative_gradient(self, y, pred):
        return np.sign(y - pred)

    def init_estimator(self):
        class MedianEstimator:
            def fit(self, y):
                self.median = np.median(y)

            def predict(self, X):
                return np.full(X.shape[0], self.median)

        return MedianEstimator()


class HuberLoss(LossFunction):
    """Huber损失"""

    def __init__(self, alpha=0.9):
        self.alpha = alpha
        self.delta = None

    def __call__(self, y, pred):
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
        if self.delta is None:
            self.delta = np.median(np.abs(y - pred))

        diff = y - pred
        mask = np.abs(diff) <= self.delta

        return np.where(mask, diff, self.delta * np.sign(diff))

    def init_estimator(self):
        class MeanEstimator:
            def fit(self, y):
                self.mean = np.mean(y)

            def predict(self, X):
                return np.full(X.shape[0], self.mean)

        return MeanEstimator()

class GradientBoostingClassifier(BaseEstimator, ClassifierMixin):
    """梯度提升分类树"""

    def __init__(self,
                 loss='deviance',  # 'deviance', 'exponential'
                 learning_rate=0.1,
                 n_estimators=100,
                 subsample=1.0,
                 criterion='friedman_mse',
                 min_samples_split=2,
                 min_samples_leaf=1,
                 max_depth=3,
                 init=None,
                 random_state=None,
                 max_features=None,
                 verbose=0):

        self.loss = loss
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.subsample = subsample
        self.criterion = criterion
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth
        self.init = init
        self.random_state = random_state
        self.max_features = max_features
        self.verbose = verbose

        if random_state is not None:
            np.random.seed(random_state)

        self.estimators_ = []
        self.train_score_ = []
        self.init_ = None
        self.classes_ = None
        self.n_classes_ = None

    def fit(self, X, y, sample_weight=None):
        """训练梯度提升分类器"""
        X, y = check_X_y(X, y)

        # 获取类别信息
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)

        if self.n_classes_ == 2:
            # 二分类
            y = np.where(y == self.classes_[0], 0, 1)
            self._fit_binary(X, y, sample_weight)
        else:
            # 多分类：使用One-vs-All
            self._fit_multiclass(X, y, sample_weight)

        return self

    def _fit_binary(self, X, y, sample_weight):
        """训练二分类模型"""
        n_samples = X.shape[0]

        # 初始化损失函数
        if self.loss == 'deviance':
            self.loss_ = BinomialDeviance()
        elif self.loss == 'exponential':
            self.loss_ = ExponentialLoss()
        else:
            raise ValueError(f"未知损失函数: {self.loss}")

        # 初始预测（对数几率）
        self.init_ = self.loss_.init_estimator()
        self.init_.fit(y)
        y_pred = self.init_.predict(np.zeros(n_samples))

        # 主循环
        self.estimators_ = []
        self.train_score_ = []

        for t in range(self.n_estimators):
            # 1. 计算负梯度
            negative_gradient = self.loss_.negative_gradient(y, y_pred)

            # 2. 子采样
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

            # 3. 训练决策树拟合负梯度
            tree = DecisionTreeRegressor(
                criterion=self.criterion,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                random_state=self.random_state
            )

            tree.fit(X_subset, y_subset, sample_weight=sample_weight_subset)
            self.estimators_.append(tree)

            # 4. 更新预测
            update = tree.predict(X)
            y_pred += self.learning_rate * update

            # 5. 记录训练损失
            self.train_score_.append(self.loss_(y, y_pred))

            if self.verbose > 0 and t % 10 == 0:
                print(f"Iteration {t}, loss = {self.train_score_[-1]:.4f}")

    def _fit_multiclass(self, X, y, sample_weight):
        """训练多分类模型"""
        n_samples = X.shape[0]

        # 转换为one-hot编码
        y_onehot = np.eye(self.n_classes_)[y]

        # 初始化损失函数（多项偏差）
        self.loss_ = MultinomialDeviance(self.n_classes_)

        # 初始预测
        self.init_ = self.loss_.init_estimator()
        self.init_.fit(y_onehot)
        y_pred = self.init_.predict(np.zeros((n_samples, 1)))

        # 为每个类别维护一个提升模型
        self.estimators_ = [[] for _ in range(self.n_classes_)]

        # 主循环
        self.train_score_ = []

        for t in range(self.n_estimators):
            for k in range(self.n_classes_):
                # 计算第k类的负梯度
                negative_gradient = self.loss_.negative_gradient(
                    y_onehot[:, k], y_pred[:, k], k
                )

                # 子采样
                if self.subsample < 1.0:
                    sample_mask = np.random.rand(n_samples) < self.subsample
                    X_subset = X[sample_mask]
                    y_subset = negative_gradient[sample_mask]
                else:
                    X_subset = X
                    y_subset = negative_gradient

                # 训练决策树
                tree = DecisionTreeRegressor(
                    criterion=self.criterion,
                    max_depth=self.max_depth,
                    min_samples_split=self.min_samples_split,
                    min_samples_leaf=self.min_samples_leaf,
                    max_features=self.max_features,
                    random_state=self.random_state
                )

                tree.fit(X_subset, y_subset)
                self.estimators_[k].append(tree)

                # 更新预测
                update = tree.predict(X)
                y_pred[:, k] += self.learning_rate * update

            # 记录训练损失
            self.train_score_.append(self.loss_(y_onehot, y_pred))

            if self.verbose > 0 and t % 10 == 0:
                print(f"Iteration {t}, loss = {self.train_score_[-1]:.4f}")

    def predict(self, X):
        """预测类别"""
        check_is_fitted(self)
        X = check_array(X)

        if self.n_classes_ == 2:
            # 二分类
            proba = self.predict_proba(X)
            return np.where(proba[:, 1] > 0.5, self.classes_[1], self.classes_[0])
        else:
            # 多分类
            proba = self.predict_proba(X)
            return self.classes_[np.argmax(proba, axis=1)]

    def predict_proba(self, X):
        """预测概率"""
        check_is_fitted(self)
        X = check_array(X)

        if self.n_classes_ == 2:
            # 二分类：使用sigmoid函数
            raw_pred = self._raw_predict(X)
            proba = 1.0 / (1.0 + np.exp(-raw_pred))
            return np.column_stack([1 - proba, proba])
        else:
            # 多分类：使用softmax
            raw_pred = self._raw_predict(X)
            exp_pred = np.exp(raw_pred - np.max(raw_pred, axis=1, keepdims=True))
            return exp_pred / np.sum(exp_pred, axis=1, keepdims=True)

    def _raw_predict(self, X):
        """原始预测（对数几率）"""
        n_samples = X.shape[0]

        if self.n_classes_ == 2:
            # 二分类
            raw_pred = self.init_.predict(np.zeros(n_samples))
            for tree in self.estimators_:
                raw_pred += self.learning_rate * tree.predict(X)
            return raw_pred
        else:
            # 多分类
            raw_pred = np.zeros((n_samples, self.n_classes_))
            for k in range(self.n_classes_):
                raw_pred[:, k] = self.init_.predict(np.zeros(n_samples))
                for tree in self.estimators_[k]:
                    raw_pred[:, k] += self.learning_rate * tree.predict(X)
            return raw_pred

    def staged_predict_proba(self, X):
        """按阶段预测概率"""
        check_is_fitted(self)
        X = check_array(X)

        n_samples = X.shape[0]

        if self.n_classes_ == 2:
            # 二分类
            raw_pred = self.init_.predict(np.zeros(n_samples))
            yield self._sigmoid_proba(raw_pred)

            for tree in self.estimators_:
                raw_pred += self.learning_rate * tree.predict(X)
                yield self._sigmoid_proba(raw_pred)
        else:
            # 多分类
            raw_pred = np.zeros((n_samples, self.n_classes_))
            for k in range(self.n_classes_):
                raw_pred[:, k] = self.init_.predict(np.zeros(n_samples))

            yield self._softmax_proba(raw_pred)

            for t in range(len(self.estimators_[0])):
                for k in range(self.n_classes_):
                    raw_pred[:, k] += (self.learning_rate *
                                     self.estimators_[k][t].predict(X))
                yield self._softmax_proba(raw_pred)

    def _sigmoid_proba(self, raw_pred):
        """sigmoid转换"""
        proba = 1.0 / (1.0 + np.exp(-raw_pred))
        return np.column_stack([1 - proba, proba])

    def _softmax_proba(self, raw_pred):
        """softmax转换"""
        exp_pred = np.exp(raw_pred - np.max(raw_pred, axis=1, keepdims=True))
        return exp_pred / np.sum(exp_pred, axis=1, keepdims=True)

    def score(self, X, y):
        """计算准确率"""
        y_pred = self.predict(X)
        return np.mean(y_pred == y)

class BinomialDeviance(LossFunction):
    """二项偏差损失（对数似然损失）"""

    def __call__(self, y, pred):
        # pred是对数几率
        return np.mean(np.log(1 + np.exp(-(2*y - 1) * pred)))

    def negative_gradient(self, y, pred):
        # 负梯度 = y - σ(pred)
        prob = 1.0 / (1.0 + np.exp(-pred))
        return y - prob

    def init_estimator(self):
        class LogOddsEstimator:
            def fit(self, y):
                # 初始化为对数几率
                pos = np.mean(y)
                if pos <= 0 or pos >= 1:
                    pos = np.clip(pos, 1e-10, 1 - 1e-10)
                self.prior = np.log(pos / (1 - pos))

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

            def predict(self, X):
                return np.full(X.shape[0], self.constant)

        return ZeroEstimator()


class MultinomialDeviance(LossFunction):
    """多项偏差损失（多分类对数似然）"""

    def __init__(self, n_classes):
        self.n_classes = n_classes

    def __call__(self, y, pred):
        # y: one-hot编码, pred: 每个类别的对数几率
        # 计算softmax概率
        exp_pred = np.exp(pred - np.max(pred, axis=1, keepdims=True))
        prob = exp_pred / np.sum(exp_pred, axis=1, keepdims=True)

        # 对数似然损失
        log_likelihood = np.sum(y * np.log(prob + 1e-15))
        return -log_likelihood / len(y)

    def negative_gradient(self, y, pred, k=None):
        # 对于多分类，每个类别单独计算
        if k is not None:
            # 计算第k类的负梯度
            exp_pred = np.exp(pred - np.max(pred, axis=1, keepdims=True))
            prob = exp_pred / np.sum(exp_pred, axis=1, keepdims=True)
            return y - prob
        else:
            # 返回所有类别的负梯度
            exp_pred = np.exp(pred - np.max(pred, axis=1, keepdims=True))
            prob = exp_pred / np.sum(exp_pred, axis=1, keepdims=True)
            return y - prob

    def init_estimator(self):
        class ZeroEstimator:
            def fit(self, y):
                # 多分类初始化为0
                self.constant = 0.0

            def predict(self, X):
                if len(X.shape) == 1:
                    return np.full(X.shape[0], self.constant)
                else:
                    return np.full((X.shape[0], 1), self.constant)

        return ZeroEstimator()
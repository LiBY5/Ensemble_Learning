"""Boosting算法实现"""  # 模块文档字符串
import numpy as np  # 导入numpy数值计算库
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin  # 导入scikit-learn基类
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor  # 导入决策树作为基学习器
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted  # 导入验证工具
from copy import deepcopy  # 导入深拷贝工具


class AdaBoostClassifier(BaseEstimator, ClassifierMixin):
    """AdaBoost分类器"""  # 类文档字符串

    def __init__(self,
                 base_estimator=None,  # 基础学习器，默认使用决策树桩
                 n_estimators=50,  # 集成器数量（弱学习器数量）
                 learning_rate=1.0,  # 学习率，控制每个弱学习器的贡献
                 algorithm='SAMME',  # 算法类型：SAMME或SAMME.R
                 random_state=None):  # 随机种子
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.algorithm = algorithm
        self.random_state = random_state

        if self.base_estimator is None:  # 如果没有指定基础学习器
            self.base_estimator = DecisionTreeClassifier(max_depth=1)  # 使用深度为1的决策树（决策树桩）

        if random_state is not None:  # 如果指定了随机种子
            np.random.seed(random_state)  # 设置numpy随机种子

        self.estimators_ = []  # 存储训练好的弱学习器列表
        self.estimator_weights_ = np.zeros(self.n_estimators)  # 存储每个弱学习器的权重
        self.estimator_errors_ = np.zeros(self.n_estimators)  # 存储每个弱学习器的错误率
        self.classes_ = None  # 存储类别标签

    def fit(self, X, y, sample_weight=None):  # 训练方法，X为特征，y为标签，sample_weight为样本权重
        X, y = check_X_y(X, y)  # 检查X和y的格式和有效性
        n_samples, n_features = X.shape  # 获取样本数和特征数

        self.classes_ = np.unique(y)  # 获取所有唯一的类别标签
        n_classes = len(self.classes_)  # 获取类别数量

        if n_classes == 2:  # 如果是二分类问题
            y_coded = np.where(y == self.classes_[0], -1, 1)  # 将标签编码为-1和1
            self._binary = True  # 标记为二分类
        else:  # 如果是多分类问题
            y_coded = np.eye(n_classes)[y]  # 使用one-hot编码
            self._binary = False  # 标记为多分类

        if sample_weight is None:  # 如果没有提供样本权重
            sample_weight = np.ones(n_samples) / n_samples  # 初始化所有样本权重相同
        else:  # 如果提供了样本权重
            sample_weight = np.array(sample_weight) / np.sum(sample_weight)  # 归一化样本权重

        self.estimators_ = []  # 重置弱学习器列表
        self.estimator_weights_ = np.zeros(self.n_estimators)  # 重置权重数组
        self.estimator_errors_ = np.zeros(self.n_estimators)  # 重置错误率数组

        for t in range(self.n_estimators):  # 遍历每个弱学习器
            print(f"\n第 {t + 1}/{self.n_estimators} 轮训练...")  # 打印训练进度

            estimator = self._make_estimator()  # 创建新的弱学习器实例

            if self._binary:  # 如果是二分类
                estimator.fit(X, y_coded, sample_weight=sample_weight)  # 使用编码后的标签训练
                y_pred = estimator.predict(X)  # 预测训练集
            else:  # 如果是多分类
                estimator.fit(X, y, sample_weight=sample_weight)  # 使用原始标签训练
                y_pred = estimator.predict(X)  # 预测训练集
                y_pred_coded = np.eye(n_classes)[np.searchsorted(self.classes_, y_pred)]  # 将预测转换为one-hot编码

            if self._binary:  # 如果是二分类
                incorrect = (y_pred != y_coded)  # 计算哪些样本预测错误
                estimator_error = np.dot(sample_weight, incorrect)  # 计算加权错误率
            else:  # 如果是多分类
                if self.algorithm == 'SAMME':  # 如果使用SAMME算法
                    incorrect = (y_pred != y)  # 计算哪些样本预测错误
                    estimator_error = np.dot(sample_weight, incorrect)  # 计算加权错误率
                else:  # 如果使用SAMME.R算法
                    y_proba = estimator.predict_proba(X)  # 获取预测概率
                    y_proba = np.clip(y_proba, 1e-15, 1 - 1e-15)  # 裁剪概率值避免数值问题
                    log_proba = np.log(y_proba)  # 计算对数概率
                    inner_product = np.sum(y_coded * log_proba, axis=1)  # 计算内积
                    estimator_error = np.dot(sample_weight, 1 - inner_product)  # 计算加权错误率

            print(f"  加权错误率: {estimator_error:.4f}")  # 打印错误率

            if estimator_error >= 0.5:  # 如果错误率大于等于0.5（弱学习器比随机猜测还差）
                print(f"  错误率≥0.5，提前停止训练")  # 打印停止信息
                self.n_estimators = t  # 更新弱学习器数量
                self.estimator_weights_ = self.estimator_weights_[:t]  # 截断权重数组
                self.estimator_errors_ = self.estimator_errors_[:t]  # 截断错误率数组
                break  # 停止训练

            if estimator_error <= 0:  # 如果错误率为0（完美分类）
                print(f"  错误率为0，完美分类，提前停止")  # 打印停止信息
                self.estimator_weights_[t] = 1.0  # 设置权重为1
                self.estimator_errors_[t] = 0  # 设置错误率为0
                self.estimators_.append(estimator)  # 将学习器添加到列表
                self.n_estimators = t + 1  # 更新弱学习器数量
                break  # 停止训练

            if self.algorithm == 'SAMME' or self._binary:  # 如果是SAMME算法或二分类
                if self._binary:  # 如果是二分类
                    estimator_weight = self.learning_rate * 0.5 * np.log(
                        (1 - estimator_error) / estimator_error  # 计算弱学习器权重（二分类公式）
                    )
                else:  # 如果是多分类SAMME
                    estimator_weight = self.learning_rate * (
                            0.5 * np.log((1 - estimator_error) / estimator_error) +  # 计算弱学习器权重（多分类公式）
                            np.log(n_classes - 1)  # 多分类调整项
                    )
            else:  # 如果是SAMME.R算法
                estimator_weight = 1.0  # SAMME.R中权重固定为1

            self.estimator_weights_[t] = estimator_weight  # 保存权重
            self.estimator_errors_[t] = estimator_error  # 保存错误率
            self.estimators_.append(estimator)  # 将学习器添加到列表

            print(f"  基学习器权重: {estimator_weight:.4f}")  # 打印权重

            if self.algorithm == 'SAMME' or self._binary:  # 如果是SAMME算法或二分类
                if self._binary:  # 如果是二分类
                    sample_weight *= np.exp(
                        -estimator_weight * y_coded * y_pred  # 更新样本权重（二分类公式）
                    )
                else:  # 如果是多分类SAMME
                    incorrect = (y_pred != y)  # 计算错误样本
                    sample_weight *= np.exp(
                        estimator_weight * incorrect  # 更新样本权重（多分类公式）
                    )
            else:  # 如果是SAMME.R算法
                sample_weight *= np.exp(
                    -((n_classes - 1) / n_classes) *  # 更新样本权重（SAMME.R公式）
                    np.sum(y_coded * np.log(y_proba), axis=1)
                )

            sample_weight /= np.sum(sample_weight)  # 归一化样本权重

            print(f"  样本权重范围: [{sample_weight.min():.6f}, {sample_weight.max():.6f}]")  # 打印权重范围
            print(f"  平均样本权重: {sample_weight.mean():.6f}")  # 打印平均权重

        return self  # 返回self以支持链式调用

    def _make_estimator(self):  # 创建弱学习器实例的辅助方法
        return deepcopy(self.base_estimator)  # 深拷贝基础学习器模板

    def predict(self, X):  # 预测方法
        check_is_fitted(self)  # 检查模型是否已训练
        X = check_array(X)  # 检查输入X的有效性

        if self._binary:  # 如果是二分类
            pred = self.decision_function(X)  # 获取决策函数值
            return np.where(pred > 0, self.classes_[1], self.classes_[0])  # 根据决策函数值预测类别
        else:  # 如果是多分类
            n_samples = X.shape[0]  # 获取样本数
            n_classes = len(self.classes_)  # 获取类别数

            pred = np.zeros((n_samples, n_classes))  # 初始化预测矩阵

            for estimator, weight in zip(self.estimators_, self.estimator_weights_):  # 遍历所有弱学习器
                if self.algorithm == 'SAMME':  # 如果是SAMME算法
                    y_pred = estimator.predict(X)  # 获取预测标签
                    pred[np.arange(n_samples),
                    np.searchsorted(self.classes_, y_pred)] += weight  # 累加权重到对应类别
                else:  # 如果是SAMME.R算法
                    y_proba = estimator.predict_proba(X)  # 获取预测概率
                    pred += weight * y_proba  # 加权累加概率

            return self.classes_[np.argmax(pred, axis=1)]  # 返回概率最大的类别

    def decision_function(self, X):  # 决策函数（仅用于二分类）
        if not self._binary:  # 如果不是二分类
            raise ValueError("decision_function仅适用于二分类")  # 抛出错误

        check_is_fitted(self)  # 检查模型是否已训练
        X = check_array(X)  # 检查输入X的有效性

        pred = np.zeros(X.shape[0])  # 初始化预测数组

        for estimator, weight in zip(self.estimators_, self.estimator_weights_):  # 遍历所有弱学习器
            y_pred = estimator.predict(X)  # 获取预测结果
            pred += weight * y_pred  # 加权累加

        return pred  # 返回决策函数值

    def staged_predict(self, X):  # 阶段预测生成器
        check_is_fitted(self)  # 检查模型是否已训练
        X = check_array(X)  # 检查输入X的有效性

        n_samples = X.shape[0]  # 获取样本数

        if self._binary:  # 如果是二分类
            for t in range(1, self.n_estimators + 1):  # 遍历每个阶段
                pred = np.zeros(n_samples)  # 初始化预测数组
                for estimator, weight in zip(self.estimators_[:t],
                                             self.estimator_weights_[:t]):  # 遍历前t个学习器
                    y_pred = estimator.predict(X)  # 获取预测结果
                    pred += weight * y_pred  # 加权累加
                yield np.where(pred > 0, self.classes_[1], self.classes_[0])  # 生成预测结果
        else:  # 如果是多分类
            n_classes = len(self.classes_)  # 获取类别数
            for t in range(1, self.n_estimators + 1):  # 遍历每个阶段
                pred = np.zeros((n_samples, n_classes))  # 初始化预测矩阵
                for estimator, weight in zip(self.estimators_[:t],
                                             self.estimator_weights_[:t]):  # 遍历前t个学习器
                    if self.algorithm == 'SAMME':  # 如果是SAMME算法
                        y_pred = estimator.predict(X)  # 获取预测标签
                        pred[np.arange(n_samples),
                        np.searchsorted(self.classes_, y_pred)] += weight  # 累加权重
                    else:  # 如果是SAMME.R算法
                        y_proba = estimator.predict_proba(X)  # 获取预测概率
                        pred += weight * y_proba  # 加权累加概率
                yield self.classes_[np.argmax(pred, axis=1)]  # 生成预测结果

    def score(self, X, y):  # 计算准确率
        y_pred = self.predict(X)  # 预测
        return np.mean(y_pred == y)  # 返回准确率


class AdaBoostRegressor(BaseEstimator, RegressorMixin):
    """AdaBoost回归器"""  # 类文档字符串

    def __init__(self, base_estimator=None, n_estimators=50,
                 learning_rate=1.0, loss='linear', random_state=None):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.loss = loss
        self.random_state = random_state

        if self.base_estimator is None:  # 如果没有指定基础学习器
            self.base_estimator = DecisionTreeRegressor(max_depth=3)  # 使用深度为3的决策树

        if random_state is not None:  # 如果指定了随机种子
            np.random.seed(random_state)  # 设置numpy随机种子

    def fit(self, X, y, sample_weight=None):  # 训练方法
        X, y = check_X_y(X, y)  # 检查X和y的格式和有效性
        n_samples = X.shape[0]  # 获取样本数

        if sample_weight is None:  # 如果没有提供样本权重
            sample_weight = np.ones(n_samples) / n_samples  # 初始化所有样本权重相同

        self.estimators_ = []  # 存储训练好的弱学习器列表
        self.estimator_weights_ = np.zeros(self.n_estimators)  # 存储每个弱学习器的权重
        self.estimator_errors_ = np.zeros(self.n_estimators)  # 存储每个弱学习器的误差

        y_pred = np.zeros(n_samples)  # 初始化预测值

        for t in range(self.n_estimators):  # 遍历每个弱学习器
            print(f"\n第 {t + 1}/{self.n_estimators} 轮回归训练...")  # 打印训练进度

            estimator = deepcopy(self.base_estimator)  # 创建新的弱学习器实例
            estimator.fit(X, y, sample_weight=sample_weight)  # 训练弱学习器
            y_pred_i = estimator.predict(X)  # 预测训练集

            error_vector = y - (y_pred + y_pred_i)  # 计算误差向量

            if self.loss == 'linear':  # 如果是线性损失
                loss_vector = np.abs(error_vector)  # 绝对误差
            elif self.loss == 'square':  # 如果是平方损失
                loss_vector = error_vector ** 2  # 平方误差
            elif self.loss == 'exponential':  # 如果是指数损失
                loss_vector = 1 - np.exp(-np.abs(error_vector))  # 指数误差

            estimator_error = np.dot(sample_weight, loss_vector)  # 计算加权损失
            self.estimator_errors_[t] = estimator_error  # 保存损失
            print(f"  加权损失: {estimator_error:.4f}")  # 打印损失

            if estimator_error >= 1.0:  # 如果损失大于等于1.0
                print(f"  损失≥1.0，提前停止")  # 打印停止信息
                break  # 停止训练

            estimator_weight = self.learning_rate * np.log(
                (1 - estimator_error) / estimator_error  # 计算弱学习器权重
            )
            self.estimator_weights_[t] = estimator_weight  # 保存权重
            self.estimators_.append(estimator)  # 将学习器添加到列表

            print(f"  基学习器权重: {estimator_weight:.4f}")  # 打印权重

            sample_weight *= np.exp(estimator_weight * loss_vector)  # 更新样本权重
            sample_weight /= sample_weight.sum()  # 归一化样本权重

            y_pred += estimator_weight * y_pred_i  # 更新累积预测值

        return self  # 返回self以支持链式调用

    def predict(self, X):  # 预测方法
        check_is_fitted(self)  # 检查模型是否已训练
        X = check_array(X)  # 检查输入X的有效性

        y_pred = np.zeros(X.shape[0])  # 初始化预测数组
        for estimator, weight in zip(self.estimators_, self.estimator_weights_):  # 遍历所有弱学习器
            y_pred += weight * estimator.predict(X)  # 加权累加预测结果

        return y_pred  # 返回最终预测


# ==================== 梯度提升相关类 ==================== #

class LossFunction:
    """损失函数基类"""  # 类文档字符串

    def __init__(self):
        pass

    def __call__(self, y, pred):  # 计算损失
        raise NotImplementedError  # 抽象方法，子类必须实现

    def negative_gradient(self, y, pred):  # 计算负梯度（伪残差）
        raise NotImplementedError  # 抽象方法，子类必须实现

    def init_estimator(self):  # 初始化估计器
        raise NotImplementedError  # 抽象方法，子类必须实现


class LeastSquaresError(LossFunction):
    """平方损失函数（用于回归）"""  # 类文档字符串

    def __call__(self, y, pred):  # 计算平方损失
        return np.mean((y - pred) ** 2)  # 均方误差

    def negative_gradient(self, y, pred):  # 计算负梯度（残差）
        return y - pred  # 残差 = 真实值 - 预测值

    def init_estimator(self):  # 初始化估计器（均值估计器）
        class MeanEstimator:
            def fit(self, y):
                self.mean = np.mean(y)  # 计算均值
                return self

            def predict(self, X):
                return np.full(X.shape[0], self.mean)  # 返回均值作为预测
        return MeanEstimator()


class LeastAbsoluteError(LossFunction):
    """绝对损失（用于回归）"""  # 类文档字符串

    def __call__(self, y, pred):  # 计算绝对损失
        return np.mean(np.abs(y - pred))  # 平均绝对误差

    def negative_gradient(self, y, pred):  # 计算负梯度（符号函数）
        return np.sign(y - pred)  # 符号函数

    def init_estimator(self):  # 初始化估计器（中位数估计器）
        class MedianEstimator:
            def fit(self, y):
                self.median = np.median(y)  # 计算中位数
                return self

            def predict(self, X):
                return np.full(X.shape[0], self.median)  # 返回中位数作为预测
        return MedianEstimator()


class HuberLoss(LossFunction):
    """Huber损失函数（对异常值鲁棒）"""  # 类文档字符串

    def __init__(self, alpha=0.9):
        self.alpha = alpha  # 参数alpha
        self.delta = None  # 阈值delta，初始为None

    def __call__(self, y, pred):  # 计算Huber损失
        diff = y - pred  # 计算残差

        if self.delta is None:  # 如果delta未初始化
            self.delta = np.median(np.abs(diff))  # 使用残差绝对值的中位数初始化delta

        mask = np.abs(diff) <= self.delta  # 创建掩码：绝对残差小于等于delta的样本
        loss = np.where(mask,
                        0.5 * diff ** 2,  # 平方损失区域
                        self.delta * (np.abs(diff) - 0.5 * self.delta))  # 线性损失区域

        return np.mean(loss)  # 返回平均损失

    def negative_gradient(self, y, pred):  # 计算负梯度
        if self.delta is None:  # 如果delta未初始化
            self.delta = np.median(np.abs(y - pred))  # 初始化delta

        diff = y - pred  # 计算残差
        mask = np.abs(diff) <= self.delta  # 创建掩码

        return np.where(mask, diff, self.delta * np.sign(diff))  # 根据区域返回梯度

    def init_estimator(self):  # 初始化估计器（均值估计器）
        class MeanEstimator:
            def fit(self, y):
                self.mean = np.mean(y)  # 计算均值
                return self

            def predict(self, X):
                return np.full(X.shape[0], self.mean)  # 返回均值作为预测
        return MeanEstimator()


class QuantileLoss(LossFunction):
    """分位数损失（用于分位数回归）"""  # 类文档字符串

    def __init__(self, alpha=0.5):
        self.alpha = alpha  # 分位数，默认中位数

    def __call__(self, y, pred):  # 计算分位数损失
        error = y - pred  # 计算残差
        loss = np.where(error > 0,
                        self.alpha * error,  # 正残差的损失
                        (self.alpha - 1) * error)  # 负残差的损失
        return np.mean(loss)  # 返回平均损失

    def negative_gradient(self, y, pred):  # 计算负梯度
        error = y - pred  # 计算残差
        return np.where(error > 0, self.alpha, self.alpha - 1)  # 返回梯度

    def init_estimator(self):  # 初始化估计器（分位数估计器）
        class QuantileEstimator:
            def __init__(self, alpha=0.5):
                self.alpha = alpha

            def fit(self, y):
                self.quantile = np.percentile(y, self.alpha * 100)  # 计算分位数
                return self

            def predict(self, X):
                return np.full(X.shape[0], self.quantile)  # 返回分位数作为预测
        return QuantileEstimator(self.alpha)


class GradientBoostingRegressor(BaseEstimator, RegressorMixin):
    """梯度提升回归树（优化版本）"""  # 类文档字符串

    def __init__(self,
                 loss='ls',  # 损失函数类型：'ls'(平方), 'lad'(绝对), 'huber', 'quantile'
                 learning_rate=0.1,  # 学习率
                 n_estimators=100,  # 弱学习器数量
                 max_depth=3,  # 决策树最大深度
                 min_samples_split=2,  # 内部节点再划分所需最小样本数
                 min_samples_leaf=1,  # 叶节点最小样本数
                 subsample=1.0,  # 子采样比例
                 max_features=None,  # 最大特征数
                 random_state=None,  # 随机种子
                 verbose=0,  # 日志详细程度
                 alpha=0.9):  # Huber损失和Quantile损失的参数

        self.loss = loss
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.subsample = subsample
        self.max_features = max_features
        self.random_state = random_state
        self.verbose = verbose
        self.alpha = alpha

        if random_state is not None:  # 如果指定了随机种子
            np.random.seed(random_state)  # 设置numpy随机种子

        self.estimators_ = []  # 存储训练好的弱学习器列表
        self.train_score_ = []  # 存储训练过程中的损失值
        self.init_ = None  # 初始化估计器
        self.loss_ = None  # 损失函数对象

    def _init_loss(self):  # 初始化损失函数
        """初始化损失函数"""
        if self.loss == 'ls':  # 如果是平方损失
            self.loss_ = LeastSquaresError()
        elif self.loss == 'lad':  # 如果是绝对损失
            self.loss_ = LeastAbsoluteError()
        elif self.loss == 'huber':  # 如果是Huber损失
            self.loss_ = HuberLoss(self.alpha)
        elif self.loss == 'quantile':  # 如果是分位数损失
            self.loss_ = QuantileLoss(self.alpha)
        else:
            raise ValueError(f"不支持的损失函数: {self.loss}")

    def _init_constant(self, y):  # 用常数初始化预测
        """用常数初始化预测"""
        self.init_ = self.loss_.init_estimator()  # 获取初始化估计器
        self.init_.fit(y)  # 拟合初始化估计器
        return self.init_.predict(np.zeros(len(y)))  # 返回初始预测

    def fit(self, X, y, sample_weight=None):  # 训练梯度提升模型
        """训练梯度提升模型"""
        X, y = check_X_y(X, y)  # 检查输入数据
        n_samples, n_features = X.shape  # 获取数据形状

        if self.verbose > 0:  # 如果启用了详细输出
            print("=" * 60)
            print("开始训练梯度提升回归树")
            print("=" * 60)
            print(f"样本数: {n_samples}, 特征数: {n_features}")
            print(f"参数: loss={self.loss}, learning_rate={self.learning_rate}")
            print(f"      n_estimators={self.n_estimators}, max_depth={self.max_depth}")

        self._init_loss()  # 初始化损失函数

        y_pred = self._init_constant(y)  # 获取初始预测

        if self.verbose > 0 and hasattr(self.init_, 'mean'):  # 如果启用了详细输出且初始化器有mean属性
            print(f"初始预测（常数）: {self.init_.mean:.4f}")

        self.estimators_ = []  # 重置弱学习器列表
        self.train_score_ = []  # 重置训练损失列表

        initial_loss = self.loss_(y, y_pred)  # 计算初始损失
        self.train_score_.append(initial_loss)  # 保存初始损失

        if self.verbose > 0:  # 如果启用了详细输出
            print(f"初始损失: {initial_loss:.4f}")

        for t in range(self.n_estimators):  # 遍历每个弱学习器
            negative_gradient = self.loss_.negative_gradient(y, y_pred)  # 计算负梯度（伪残差）

            if self.subsample < 1.0:  # 如果使用了子采样
                sample_mask = np.random.rand(n_samples) < self.subsample  # 创建采样掩码
                X_subset = X[sample_mask]  # 子采样特征
                y_subset = negative_gradient[sample_mask]  # 子采样梯度
                sample_weight_subset = (sample_weight[sample_mask]
                                       if sample_weight is not None else None)  # 子采样权重
            else:  # 如果不使用子采样
                X_subset = X
                y_subset = negative_gradient
                sample_weight_subset = sample_weight

            tree = DecisionTreeRegressor(  # 创建决策树回归器
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                random_state=self.random_state
            )

            tree.fit(X_subset, y_subset, sample_weight=sample_weight_subset)  # 训练决策树拟合负梯度
            self.estimators_.append(tree)  # 保存决策树

            update = tree.predict(X)  # 获取决策树预测
            y_pred += self.learning_rate * update  # 更新累积预测值

            current_loss = self.loss_(y, y_pred)  # 计算当前损失
            self.train_score_.append(current_loss)  # 保存当前损失

            if self.verbose > 0 and t % 10 == 0:  # 如果启用了详细输出且每10轮输出一次
                print(f"轮次 {t + 1:3d}/{self.n_estimators}: 损失 = {current_loss:.4f}")

        if self.verbose > 0:  # 如果启用了详细输出
            print(f"训练完成，最终损失: {current_loss:.4f}")

        return self  # 返回self以支持链式调用

    def predict(self, X):  # 预测方法
        """预测"""
        check_is_fitted(self)  # 检查模型是否已训练
        X = check_array(X)  # 检查输入数据

        y_pred = self.init_.predict(np.zeros(X.shape[0]))  # 获取初始预测

        for tree in self.estimators_:  # 遍历所有弱学习器
            y_pred += self.learning_rate * tree.predict(X)  # 加权累加预测结果

        return y_pred  # 返回最终预测

    def staged_predict(self, X):  # 阶段预测生成器
        """按阶段预测"""
        check_is_fitted(self)  # 检查模型是否已训练
        X = check_array(X)  # 检查输入数据

        y_pred = self.init_.predict(np.zeros(X.shape[0]))  # 获取初始预测

        yield y_pred.copy()  # 生成初始预测

        for tree in self.estimators_:  # 遍历所有弱学习器
            y_pred += self.learning_rate * tree.predict(X)  # 更新预测
            yield y_pred.copy()  # 生成当前阶段预测


# ==================== 分类损失函数 ==================== #

class BinomialDeviance(LossFunction):
    """二项偏差损失（对数似然损失，用于二分类）"""  # 类文档字符串

    def __call__(self, y, pred):  # 计算二项偏差损失
        # y ∈ {0, 1}, pred是对数几率
        pred = np.clip(pred, -500, 500)  # 裁剪预测值避免数值溢出
        return np.mean(np.log(1 + np.exp(-(2 * y - 1) * pred)))  # 对数似然损失

    def negative_gradient(self, y, pred):  # 计算负梯度
        # 负梯度 = y - σ(pred)，其中σ是sigmoid函数
        prob = 1.0 / (1.0 + np.exp(-pred))  # 计算概率
        return y - prob  # 计算残差

    def init_estimator(self):  # 初始化估计器（对数几率估计器）
        class LogOddsEstimator:
            def fit(self, y):
                pos = np.mean(y)  # 计算正类比例
                if pos <= 0 or pos >= 1:  # 如果比例在边界上
                    pos = np.clip(pos, 1e-10, 1 - 1e-10)  # 裁剪避免数值问题
                self.prior = np.log(pos / (1 - pos))  # 计算先验对数几率
                return self

            def predict(self, X):
                return np.full(X.shape[0], self.prior)  # 返回先验对数几率
        return LogOddsEstimator()


class ExponentialLoss(LossFunction):
    """指数损失（AdaBoost损失）"""  # 类文档字符串

    def __call__(self, y, pred):  # 计算指数损失
        # 假设y ∈ {0, 1}，转换为{-1, 1}
        y_transformed = 2 * y - 1  # 转换标签
        return np.mean(np.exp(-y_transformed * pred))  # 指数损失

    def negative_gradient(self, y, pred):  # 计算负梯度
        y_transformed = 2 * y - 1  # 转换标签
        return y_transformed * np.exp(-y_transformed * pred)  # 梯度

    def init_estimator(self):  # 初始化估计器（零估计器）
        class ZeroEstimator:
            def fit(self, y):
                self.constant = 0.0  # 常数0
                return self

            def predict(self, X):
                return np.full(X.shape[0], self.constant)  # 返回0
        return ZeroEstimator()


class MultinomialDeviance(LossFunction):
    """多项偏差损失（多分类对数似然）"""  # 类文档字符串

    def __init__(self, n_classes):
        self.n_classes = n_classes  # 类别数

    def __call__(self, y, pred):  # 计算多项偏差损失
        # y: one-hot编码, pred: 每个类别的对数几率
        pred = np.clip(pred, -500, 500)  # 裁剪预测值避免数值溢出
        exp_pred = np.exp(pred - np.max(pred, axis=1, keepdims=True))  # 数值稳定的指数计算
        prob = exp_pred / np.sum(exp_pred, axis=1, keepdims=True)  # softmax概率

        log_likelihood = np.sum(y * np.log(prob + 1e-15))  # 对数似然
        return -log_likelihood / len(y)  # 负对数似然（损失）

    def negative_gradient(self, y, pred, k=None):  # 计算负梯度
        if k is not None:  # 如果指定了类别k
            pred = np.clip(pred, -500, 500)  # 裁剪预测值
            exp_pred = np.exp(pred - np.max(pred, axis=1, keepdims=True))  # 数值稳定的指数计算
            prob = exp_pred / np.sum(exp_pred, axis=1, keepdims=True)  # softmax概率
            return y[:, k] - prob[:, k]  # 类别k的梯度
        else:  # 如果未指定类别
            pred = np.clip(pred, -500, 500)  # 裁剪预测值
            exp_pred = np.exp(pred - np.max(pred, axis=1, keepdims=True))  # 数值稳定的指数计算
            prob = exp_pred / np.sum(exp_pred, axis=1, keepdims=True)  # softmax概率
            return y - prob  # 所有类别的梯度

    def init_estimator(self):  # 初始化估计器（零估计器）
        class ZeroEstimator:
            def fit(self, y):
                self.constant = 0.0  # 常数0
                return self

            def predict(self, X):
                if len(X.shape) == 1:  # 如果X是一维的
                    return np.full(X.shape[0], self.constant)  # 返回一维数组
                else:  # 如果X是二维的
                    return np.full((X.shape[0], 1), self.constant)  # 返回二维数组
        return ZeroEstimator()


class GradientBoostingClassifier(BaseEstimator, ClassifierMixin):
    """梯度提升分类树"""  # 类文档字符串

    def __init__(self,
                 loss='deviance',  # 损失函数：'deviance'(偏差), 'exponential'(指数)
                 learning_rate=0.1,  # 学习率
                 n_estimators=100,  # 弱学习器数量
                 max_depth=3,  # 决策树最大深度
                 min_samples_split=2,  # 内部节点再划分所需最小样本数
                 min_samples_leaf=1,  # 叶节点最小样本数
                 subsample=1.0,  # 子采样比例
                 max_features=None,  # 最大特征数
                 random_state=None,  # 随机种子
                 verbose=0):  # 日志详细程度

        self.loss = loss
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.subsample = subsample
        self.max_features = max_features
        self.random_state = random_state
        self.verbose = verbose

        if random_state is not None:  # 如果指定了随机种子
            np.random.seed(random_state)  # 设置numpy随机种子

        self.estimators_ = []  # 存储训练好的弱学习器列表
        self.train_score_ = []  # 存储训练过程中的损失值
        self.init_ = None  # 初始化估计器
        self.classes_ = None  # 类别标签
        self.n_classes_ = None  # 类别数量

    def _init_loss(self, n_classes):  # 初始化损失函数
        """初始化损失函数"""
        if self.loss == 'deviance':  # 如果是偏差损失
            if n_classes == 2:  # 如果是二分类
                self.loss_ = BinomialDeviance()  # 使用二项偏差损失
            else:  # 如果是多分类
                self.loss_ = MultinomialDeviance(n_classes)  # 使用多项偏差损失
        elif self.loss == 'exponential':  # 如果是指数损失
            if n_classes == 2:  # 如果是二分类
                class ExponentialLossWrapper:  # 包装器类
                    def negative_gradient(self, y, pred):
                        y_transformed = 2 * y - 1  # 转换标签
                        return 2 * y_transformed * np.exp(-2 * y_transformed * pred)  # 梯度

                    def init_estimator(self):
                        class ZeroEstimator:
                            def fit(self, y):
                                self.constant = 0.0
                                return self

                            def predict(self, X):
                                return np.full(X.shape[0], self.constant)
                        return ZeroEstimator()
                self.loss_ = ExponentialLossWrapper()  # 使用指数损失包装器
            else:
                raise ValueError("指数损失仅支持二分类")  # 多分类不支持指数损失
        else:
            raise ValueError(f"不支持的损失函数: {self.loss}")

    def fit(self, X, y, sample_weight=None):  # 训练梯度提升分类器
        """训练梯度提升分类器"""
        X, y = check_X_y(X, y)  # 检查输入数据

        self.classes_ = np.unique(y)  # 获取类别标签
        self.n_classes_ = len(self.classes_)  # 获取类别数量

        if self.verbose > 0:  # 如果启用了详细输出
            print("=" * 60)
            print("开始训练梯度提升分类器")
            print("=" * 60)
            print(f"样本数: {X.shape[0]}, 特征数: {X.shape[1]}")
            print(f"类别数: {self.n_classes_}, 类别: {self.classes_}")
            print(f"参数: loss={self.loss}, learning_rate={self.learning_rate}")

        if self.n_classes_ == 2:  # 如果是二分类
            y_coded = np.where(y == self.classes_[0], 0, 1)  # 编码为0和1
            self._fit_binary(X, y_coded, sample_weight)  # 训练二分类模型
        else:  # 如果是多分类
            y_onehot = np.eye(self.n_classes_)[y]  # one-hot编码
            self._fit_multiclass(X, y_onehot, sample_weight)  # 训练多分类模型

        return self  # 返回self以支持链式调用

    def _fit_binary(self, X, y, sample_weight):  # 训练二分类模型
        """训练二分类模型"""
        n_samples = X.shape[0]  # 获取样本数

        self._init_loss(2)  # 初始化损失函数（二分类）

        self.init_ = self.loss_.init_estimator()  # 获取初始化估计器
        self.init_.fit(y)  # 拟合初始化估计器
        y_pred = self.init_.predict(np.zeros((n_samples, 1))).flatten()  # 获取初始预测

        if self.verbose > 0 and hasattr(self.init_, 'prior'):  # 如果启用了详细输出且初始化器有prior属性
            print(f"初始先验概率: {1/(1+np.exp(-2*self.init_.prior)):.4f}")  # 打印先验概率

        self.estimators_ = []  # 重置弱学习器列表
        self.train_score_ = []  # 重置训练损失列表

        for t in range(self.n_estimators):  # 遍历每个弱学习器
            negative_gradient = self.loss_.negative_gradient(y, y_pred)  # 计算负梯度

            if self.subsample < 1.0:  # 如果使用了子采样
                subsample_mask = np.random.rand(n_samples) < self.subsample  # 创建采样掩码
                X_subset = X[subsample_mask]  # 子采样特征
                y_subset = negative_gradient[subsample_mask]  # 子采样梯度
                sample_weight_subset = None  # 不使用样本权重
            else:  # 如果不使用子采样
                X_subset = X
                y_subset = negative_gradient
                sample_weight_subset = sample_weight

            tree = DecisionTreeRegressor(  # 创建决策树回归器
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                random_state=self.random_state
            )

            tree.fit(X_subset, y_subset, sample_weight=sample_weight_subset)  # 训练决策树拟合负梯度
            self.estimators_.append(tree)  # 保存决策树

            update = tree.predict(X)  # 获取决策树预测
            y_pred += self.learning_rate * update  # 更新累积预测值

            if self.loss == 'deviance':  # 如果是偏差损失
                current_loss = np.mean(np.log(1 + np.exp(-2 * y * y_pred)))  # 计算偏差损失
            else:  # 如果是指数损失
                current_loss = np.mean(np.exp(-y * y_pred))  # 计算指数损失

            self.train_score_.append(current_loss)  # 保存当前损失

            if self.verbose > 0 and t % 10 == 0:  # 如果启用了详细输出且每10轮输出一次
                print(f"轮次 {t+1:3d}/{self.n_estimators}: 损失 = {current_loss:.4f}")

        if self.verbose > 0:  # 如果启用了详细输出
            print(f"训练完成，最终损失: {current_loss:.4f}")

    def predict(self, X):  # 预测类别
        """预测类别"""
        check_is_fitted(self)  # 检查模型是否已训练
        X = check_array(X)  # 检查输入数据

        if self.n_classes_ == 2:  # 如果是二分类
            proba = self.predict_proba(X)  # 获取预测概率
            return np.where(proba[:, 1] > 0.5, self.classes_[1], self.classes_[0])  # 根据概率阈值预测类别
        else:  # 如果是多分类
            proba = self.predict_proba(X)  # 获取预测概率
            return self.classes_[np.argmax(proba, axis=1)]  # 返回概率最大的类别

    def predict_proba(self, X):  # 预测概率
        """预测概率"""
        check_is_fitted(self)  # 检查模型是否已训练
        X = check_array(X)  # 检查输入数据

        if self.n_classes_ == 2:  # 如果是二分类
            raw_pred = self._raw_predict(X)  # 获取原始预测（对数几率）
            proba = 1.0 / (1.0 + np.exp(-raw_pred))  # sigmoid函数转换概率
            return np.column_stack([1 - proba, proba])  # 返回两类概率
        else:  # 如果是多分类
            raw_pred = self._raw_predict(X)  # 获取原始预测（对数几率）
            exp_pred = np.exp(raw_pred - np.max(raw_pred, axis=1, keepdims=True))  # 数值稳定的指数计算
            return exp_pred / np.sum(exp_pred, axis=1, keepdims=True)  # softmax转换概率

    def _raw_predict(self, X):  # 原始预测（对数几率）
        """原始预测（对数几率）"""
        n_samples = X.shape[0]  # 获取样本数

        if self.n_classes_ == 2:  # 如果是二分类
            raw_pred = self.init_.predict(np.zeros((n_samples, 1))).flatten()  # 获取初始预测
            for tree in self.estimators_:  # 遍历所有弱学习器
                raw_pred += self.learning_rate * tree.predict(X)  # 累加预测结果
            return raw_pred  # 返回原始预测
        else:  # 如果是多分类
            raw_pred = np.zeros((n_samples, self.n_classes_))  # 初始化原始预测矩阵
            for k in range(self.n_classes_):  # 遍历每个类别
                raw_pred[:, k] = self.init_.predict(np.zeros((n_samples, 1))).flatten()  # 获取初始预测
                for tree in self.estimators_[k]:  # 遍历该类别的弱学习器
                    raw_pred[:, k] += self.learning_rate * tree.predict(X)  # 累加预测结果
            return raw_pred  # 返回原始预测

    def score(self, X, y):  # 计算准确率
        """计算准确率"""
        y_pred = self.predict(X)  # 预测
        return np.mean(y_pred == y)  # 返回准确率


# 导出所有类
__all__ = [
    'AdaBoostClassifier',
    'AdaBoostRegressor',
    'GradientBoostingRegressor',
    'GradientBoostingClassifier',
    'LossFunction',
    'LeastSquaresError',
    'LeastAbsoluteError',
    'HuberLoss',
    'QuantileLoss',
    'BinomialDeviance',
    'ExponentialLoss',
    'MultinomialDeviance'
]
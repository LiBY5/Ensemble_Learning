# 导入numpy库，用于数值计算和数组操作
import numpy as np
# 导入Python的抽象基类相关模块，用于定义抽象基类和抽象方法
from abc import ABC, abstractmethod
# 从scikit-learn库导入基类和工具：BaseEstimator(所有估计器的基类)、ClassifierMixin(分类器混合类)、clone(克隆函数)
from sklearn.base import BaseEstimator, ClassifierMixin, clone


# 定义一个名为BaseModel的抽象基类，继承自ABC
class BaseModel(ABC):
    """基模型抽象类 - 定义所有模型的统一接口"""

    # 初始化方法，设置模型名称和训练状态
    def __init__(self, name="base_model"):
        # 设置模型名称，默认为"base_model"
        self.name = name
        # 标记模型是否已训练，初始化为False
        self.is_fitted = False

    # 抽象方法装饰器，表示fit方法是抽象的，子类必须实现
    @abstractmethod
    def fit(self, X, y):
        """训练模型 - 在具体子类中实现具体的训练逻辑"""
        # 抽象方法没有具体实现
        pass

    # 抽象方法装饰器，表示predict方法是抽象的，子类必须实现
    @abstractmethod
    def predict(self, X):
        """预测方法 - 在具体子类中实现具体的预测逻辑"""
        # 抽象方法没有具体实现
        pass

    # 计算模型准确率的方法
    def score(self, X, y):
        """计算准确率 - 使用模型的预测结果与真实标签比较"""
        # 动态导入accuracy_score函数，避免循环依赖
        from sklearn.metrics import accuracy_score
        # 使用模型的predict方法进行预测
        y_pred = self.predict(X)
        # 计算预测准确率并返回
        return accuracy_score(y, y_pred)

    # 特殊方法，定义对象的字符串表示形式
    def __repr__(self):
        # 返回类名和模型名称的格式化字符串
        return f"{self.__class__.__name__}(name={self.name})"


# 定义SklearnClassifier类，继承自BaseModel，用于包装scikit-learn分类器
class SklearnClassifier(BaseModel):
    """scikit-learn分类器包装器 - 将scikit-learn分类器适配到BaseModel接口"""

    # 初始化方法，接收一个scikit-learn模型实例和可选的名称参数
    def __init__(self, sklearn_model, name=None):
        # 如果未提供名称，使用底层模型的类名作为名称
        if name is None:
            name = sklearn_model.__class__.__name__
        # 调用父类的初始化方法，设置模型名称
        super().__init__(name)
        # 保存传入的scikit-learn模型实例
        self.model = sklearn_model

    # 实现fit方法，训练模型
    def fit(self, X, y, **kwargs):
        # 调用底层scikit-learn模型的fit方法进行训练
        self.model.fit(X, y, **kwargs)
        # 将训练状态标记为True
        self.is_fitted = True
        # 从标签y中提取所有唯一类别并保存，用于后续处理
        self.classes_ = np.unique(y)
        # 返回self以支持链式调用
        return self

    # 实现predict方法，进行预测
    def predict(self, X):
        # 检查模型是否已经训练，如果没有则抛出异常
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        # 调用底层模型的predict方法进行预测并返回结果
        return self.model.predict(X)

    # 实现predict_proba方法，返回预测概率
    def predict_proba(self, X):
        # 检查模型是否已经训练，如果没有则抛出异常
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        # 检查底层模型是否有predict_proba方法
        if hasattr(self.model, 'predict_proba'):
            # 如果有，直接调用底层模型的predict_proba方法
            return self.model.predict_proba(X)
        else:
            # 如果底层模型没有predict_proba方法，手动创建概率矩阵
            # 首先使用predict方法获取预测结果
            pred = self.predict(X)
            # 获取样本数量
            n_samples = len(pred)
            # 获取类别数量
            n_classes = len(self.classes_)
            # 创建全零的概率矩阵，形状为(样本数, 类别数)
            proba = np.zeros((n_samples, n_classes))

            # 遍历所有类别
            for i, cls in enumerate(self.classes_):
                # 将预测结果与当前类别比较，相等则为1，否则为0
                proba[:, i] = (pred == cls).astype(float)

            # 返回构造的概率矩阵
            return proba

    # 实现get_params方法，获取模型参数（与scikit-learn API兼容）
    def get_params(self, deep=True):
        # 调用底层模型的get_params方法
        return self.model.get_params(deep)

    # 实现set_params方法，设置模型参数（与scikit-learn API兼容）
    def set_params(self, **params):
        # 调用底层模型的set_params方法
        self.model.set_params(**params)
        # 返回self以支持链式调用
        return self


# 定义函数，获取多样化的基分类器列表
def get_diverse_classifiers(random_state=42):
    """获取多样化的基分类器列表 - 用于集成学习或模型比较"""
    # 导入scikit-learn中的各种分类器
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.naive_bayes import GaussianNB

    # 创建分类器实例列表，包含多种不同类型的分类器
    classifiers = [
        # 决策树，最大深度5
        DecisionTreeClassifier(max_depth=5, random_state=random_state),
        # 决策树，最大深度10，使用不同的随机种子
        DecisionTreeClassifier(max_depth=10, random_state=random_state+1),
        # 随机森林，10棵树，最大深度5
        RandomForestClassifier(n_estimators=10, max_depth=5, random_state=random_state),
        # 支持向量机，启用概率估计
        SVC(probability=True, random_state=random_state),
        # 逻辑回归，增加最大迭代次数确保收敛
        LogisticRegression(random_state=random_state, max_iter=1000),
        # K近邻分类器，邻居数为5
        KNeighborsClassifier(n_neighbors=5),
        # 高斯朴素贝叶斯
        GaussianNB()
    ]

    # 定义每个分类器的名称列表，与上面的分类器一一对应
    names = [
        "DecisionTree_depth5",      # 决策树（深度5）
        "DecisionTree_depth10",     # 决策树（深度10）
        "RandomForest",             # 随机森林
        "SVM",                      # 支持向量机
        "LogisticRegression",       # 逻辑回归
        "KNN",                      # K近邻
        "NaiveBayes"                # 朴素贝叶斯
    ]

    # 使用列表推导式将每个分类器包装为SklearnClassifier对象
    # zip函数将分类器实例和名称一一配对
    return [SklearnClassifier(clf, name) for clf, name in zip(classifiers, names)]


# 主程序入口，当直接运行此脚本时执行以下代码
if __name__ == "__main__":
    # 导入数据生成和分割工具
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split

    # 生成模拟分类数据集
    # 100个样本，10个特征，2个类别，固定随机种子确保可重复性
    X, y = make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=42)
    # 将数据集分割为训练集和测试集，测试集占20%
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 调用函数获取多样化的分类器列表
    classifiers = get_diverse_classifiers()

    # 打印测试标题
    print("基分类器测试:")
    # 打印分隔线
    print("-" * 50)

    # 遍历每个分类器进行训练和评估
    for clf in classifiers:
        # 使用训练数据训练分类器
        clf.fit(X_train, y_train)
        # 使用测试数据评估分类器准确率
        accuracy = clf.score(X_test, y_test)
        # 格式化输出：分类器名称（左对齐，宽度25）和准确率（保留4位小数）
        print(f"{clf.name:25s}: {accuracy:.4f}")
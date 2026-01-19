"""基模型定义和包装器"""

import numpy as np
from abc import ABC, abstractmethod
from sklearn.base import BaseEstimator, ClassifierMixin, clone


class BaseModel(ABC):
    """基模型抽象类"""

    def __init__(self, name="base_model"):
        self.name = name
        self.is_fitted = False

    @abstractmethod
    def fit(self, X, y):
        """训练模型"""
        pass

    @abstractmethod
    def predict(self, X):
        """预测"""
        pass

    def score(self, X, y):
        """计算准确率"""
        from sklearn.metrics import accuracy_score
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)

    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name})"


class SklearnClassifier(BaseModel):
    """scikit-learn分类器包装器"""

    def __init__(self, sklearn_model, name=None):
        if name is None:
            name = sklearn_model.__class__.__name__
        super().__init__(name)
        self.model = sklearn_model

    def fit(self, X, y, **kwargs):
        self.model.fit(X, y, **kwargs)
        self.is_fitted = True
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict(X)

    def predict_proba(self, X):
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        # 检查模型是否有predict_proba方法
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        else:
            # 对于没有predict_proba的模型，返回0/1概率
            pred = self.predict(X)
            n_samples = len(pred)
            n_classes = len(self.classes_)
            proba = np.zeros((n_samples, n_classes))

            for i, cls in enumerate(self.classes_):
                proba[:, i] = (pred == cls).astype(float)

            return proba

    def get_params(self, deep=True):
        return self.model.get_params(deep)

    def set_params(self, **params):
        self.model.set_params(**params)
        return self


def get_diverse_classifiers(random_state=42):
    """获取多样化的基分类器列表"""
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.naive_bayes import GaussianNB

    classifiers = [
        DecisionTreeClassifier(max_depth=5, random_state=random_state),
        DecisionTreeClassifier(max_depth=10, random_state=random_state + 1),
        RandomForestClassifier(n_estimators=10, max_depth=5, random_state=random_state),
        SVC(probability=True, random_state=random_state),
        LogisticRegression(random_state=random_state, max_iter=1000),
        KNeighborsClassifier(n_neighbors=5),
        GaussianNB()
    ]

    # 为每个分类器命名
    names = [
        "DecisionTree_depth5",
        "DecisionTree_depth10",
        "RandomForest",
        "SVM",
        "LogisticRegression",
        "KNN",
        "NaiveBayes"
    ]

    return [SklearnClassifier(clf, name) for clf, name in zip(classifiers, names)]


# 测试代码
if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split

    # 生成模拟数据
    X, y = make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 测试所有分类器
    classifiers = get_diverse_classifiers()

    print("基分类器测试:")
    print("-" * 50)

    for clf in classifiers:
        clf.fit(X_train, y_train)
        accuracy = clf.score(X_test, y_test)
        print(f"{clf.name:25s}: {accuracy:.4f}")
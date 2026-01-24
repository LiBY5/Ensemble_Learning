"""基础模型类 - models/base.py"""
import numpy as np
from abc import ABC, abstractmethod


class BaseClassifier(ABC):
    """基础分类器抽象基类"""

    def __init__(self, name="BaseClassifier"):
        """
        参数:
        ----------
        name : str
            模型名称
        """
        self.name = name
        self.is_fitted = False
        self.classes_ = None
        self.n_classes_ = None
        self.n_features_ = None

    @abstractmethod
    def fit(self, X, y):
        """训练模型"""
        pass

    @abstractmethod
    def predict(self, X):
        """预测类别"""
        pass

    def predict_proba(self, X):
        """预测概率（分类器需实现）"""
        raise NotImplementedError(f"{self.name} 必须实现 predict_proba 方法")

    def score(self, X, y):
        """计算准确率"""
        from sklearn.metrics import accuracy_score
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)

    def __repr__(self):
        """模型的字符串表示"""
        return f"{self.name}(is_fitted={self.is_fitted})"
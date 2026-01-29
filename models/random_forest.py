"""随机森林实现 - models/random_forest.py"""
import numpy as np  # 导入数值计算库
from copy import deepcopy  # 导入深拷贝函数
from typing import Optional, Union  # 导入类型提示
from sklearn.tree import DecisionTreeClassifier  # 导入决策树分类器
from sklearn.metrics import accuracy_score  # 导入准确率评估指标
import warnings  # 导入警告模块

from models.base import BaseClassifier  # 从本地导入基础分类器类

class RandomForestClassifier(BaseClassifier):
    """随机森林分类器实现"""

    def __init__(self,
                 n_estimators: int = 100,      # 森林中树的数量
                 criterion: str = 'gini',       # 分裂标准（gini/entropy）
                 max_depth: Optional[int] = None,  # 树的最大深度
                 min_samples_split: Union[int, float] = 2,  # 分裂所需最小样本数
                 min_samples_leaf: Union[int, float] = 1,   # 叶节点最小样本数
                 max_features: Union[str, float] = 'sqrt',  # 分裂时考虑的特征数
                 bootstrap: bool = True,        # 是否使用自助采样
                 oob_score: bool = False,       # 是否计算袋外分数
                 random_state: Optional[int] = None,  # 随机种子
                 n_jobs: Optional[int] = None,  # 并行作业数（未实现）
                 verbose: int = 0):             # 日志详细程度
        super().__init__(name="RandomForestClassifier")  # 调用父类构造函数

        # 树的数量和配置参数
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features

        # Bagging相关参数
        self.bootstrap = bootstrap
        self.oob_score = oob_score

        # 并行化和随机性控制
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose

        # 模型内部状态变量
        self.estimators_ = []              # 存储所有决策树
        self.estimator_features_ = []      # 存储每棵树使用的特征索引
        self.feature_importances_ = None   # 特征重要性数组
        self.oob_score_ = None             # 袋外分数

        # 设置全局随机种子
        if random_state is not None:
            np.random.seed(random_state)

    def get_params(self, deep=True):
        """获取模型参数（sklearn兼容接口）"""
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
        """设置模型参数（sklearn兼容接口）"""
        for key, value in params.items():  # 遍历所有传入参数
            setattr(self, key, value)       # 动态设置属性值
        return self  # 返回自身引用

    def fit(self, X, y):
        """训练随机森林模型"""
        # 数据格式转换与验证
        X, y = self._validate_data(X, y)  # 转换为numpy数组并验证
        n_samples, n_features = X.shape   # 获取样本数和特征数

        # 计算实际使用的特征数量
        max_features = self._compute_max_features(n_features)

        # 初始化袋外估计所需数据结构
        oob_pred_sum = None  # 存储袋外样本的预测概率和
        oob_count = None     # 存储袋外样本被预测的次数

        # 如果需要计算袋外分数
        if self.oob_score:
            n_classes = len(np.unique(y))  # 获取类别数量
            oob_pred_sum = np.zeros((n_samples, n_classes))  # 初始化概率累加矩阵
            oob_count = np.zeros(n_samples)                  # 初始化计数向量

        # 清空现有模型组件
        self.estimators_ = []          # 重置树列表
        self.estimator_features_ = []  # 重置特征索引列表

        # 训练每棵决策树
        for i in range(self.n_estimators):
            # 定期输出训练进度
            if self.verbose > 0 and i % 10 == 0:
                print(f"训练决策树 {i+1}/{self.n_estimators}")

            # 1. 样本采样（自助采样）
            if self.bootstrap:
                # 有放回随机抽样生成样本索引
                indices = np.random.choice(n_samples, size=n_samples, replace=True)
                X_sample = X[indices]      # 采样后的特征
                y_sample = y[indices]      # 采样后的标签
                sample_indices = indices   # 记录采样索引
            else:
                # 使用全部样本
                X_sample, y_sample = X, y
                sample_indices = np.arange(n_samples)  # 全索引

            # 2. 特征采样（随机特征子集）
            # 随机选择特征索引（无放回）
            feature_indices = np.random.choice(
                n_features,        # 总特征数
                size=max_features,  # 选择的特征数
                replace=False       # 无放回抽样
            )
            self.estimator_features_.append(feature_indices)  # 保存特征索引

            # 3. 训练决策树
            tree = DecisionTreeClassifier(
                criterion=self.criterion,           # 分裂标准
                max_depth=self.max_depth,           # 最大深度
                min_samples_split=self.min_samples_split,  # 分裂最小样本数
                min_samples_leaf=self.min_samples_leaf,    # 叶节点最小样本数
                random_state=self.random_state + i if self.random_state is not None else None  # 树专属随机种子
            )

            # 使用特征子集训练树
            tree.fit(X_sample[:, feature_indices], y_sample)
            self.estimators_.append(tree)  # 添加到树列表

            # 4. 更新袋外估计（如果使用自助采样且需要OOB分数）
            if self.oob_score and self.bootstrap:
                self._update_oob_estimation(
                    i,               # 当前树索引
                    X, y,            # 完整数据集
                    sample_indices,  # 本次采样的索引
                    feature_indices, # 使用的特征索引
                    oob_pred_sum,    # OOB预测累加器
                    oob_count        # OOB计数器
                )

        # 5. 计算特征重要性
        self.feature_importances_ = self._compute_feature_importances(n_features)

        # 6. 完成训练流程
        self._finalize_fit(X, y, oob_pred_sum, oob_count)

        return self  # 返回自身引用

    def predict(self, X):
        """预测类别标签（多数投票法）"""
        # 检查模型是否已训练
        if not self.is_fitted:
            raise ValueError("模型必须先训练")

        # 收集所有树的预测结果
        all_predictions = []
        for tree, feature_indices in zip(self.estimators_, self.estimator_features_):
            # 选择特征子集进行预测
            X_selected = X[:, feature_indices]
            pred = tree.predict(X_selected)  # 单棵树预测
            all_predictions.append(pred)     # 添加到结果集

        # 转换为数组 [树数量, 样本数量]
        all_predictions = np.array(all_predictions)

        # 对每个样本执行多数投票
        final_predictions = []
        for sample_idx in range(all_predictions.shape[1]):
            # 统计当前样本的投票结果
            counts = np.bincount(all_predictions[:, sample_idx], minlength=self.n_classes_)
            # 选择得票最多的类别
            final_predictions.append(np.argmax(counts))

        return np.array(final_predictions)  # 返回最终预测结果

    def predict_proba(self, X):
        """预测类别概率（平均概率法）"""
        # 检查模型是否已训练
        if not self.is_fitted:
            raise ValueError("模型必须先训练")

        # 收集所有树的概率预测
        all_probas = []
        for tree, feature_indices in zip(self.estimators_, self.estimator_features_):
            # 选择特征子集
            X_selected = X[:, feature_indices]
            # 获取概率预测 [样本数, 类别数]
            proba = tree.predict_proba(X_selected)
            all_probas.append(proba)  # 添加到结果集

        # 计算平均概率 [样本数, 类别数]
        avg_proba = np.mean(all_probas, axis=0)
        return avg_proba

    def _validate_data(self, X, y):
        """验证并转换输入数据格式"""
        X = np.array(X)  # 转换为numpy数组
        y = np.array(y)  # 转换为numpy数组

        # 检查样本数量一致性
        if len(X) != len(y):
            raise ValueError(f"X和y的长度不匹配: {len(X)} != {len(y)}")

        return X, y  # 返回转换后的数据

    def _compute_max_features(self, n_features):
        """计算每次分裂考虑的特征数量"""
        # 处理字符串类型的max_features
        if isinstance(self.max_features, str):
            if self.max_features in ['auto', 'sqrt']:  # auto和sqrt等价
                return int(np.sqrt(n_features))         # 取平方根
            elif self.max_features == 'log2':          # 对数尺度
                return int(np.log2(n_features))         # 取对数
            else:
                raise ValueError(f"不支持的max_features字符串: {self.max_features}")

        # 处理浮点数类型（特征比例）
        elif isinstance(self.max_features, float):
            if not (0.0 < self.max_features <= 1.0):  # 验证比例范围
                raise ValueError("max_features比例必须在(0, 1]范围内")
            return max(1, int(self.max_features * n_features))  # 计算特征数

        # 处理整数类型（固定数量）
        elif isinstance(self.max_features, int):
            if self.max_features <= 0:  # 验证正整数
                raise ValueError("max_features必须为正整数")
            return min(self.max_features, n_features)  # 不超过总特征数

        # 无效类型处理
        else:
            raise TypeError("max_features必须是字符串、浮点数或整数")

    def _update_oob_estimation(self, tree_idx, X, y, sample_indices,
                             feature_indices, oob_pred_sum, oob_count):
        """更新袋外样本的预测估计"""
        n_samples = X.shape[0]  # 总样本数

        # 确定袋外样本索引（未被当前树采样的样本）
        all_indices = set(range(n_samples))          # 所有样本索引集合
        in_bag_indices = set(sample_indices)         # 袋内样本索引集合
        oob_indices = list(all_indices - in_bag_indices)  # 袋外样本索引列表

        # 如果没有袋外样本则跳过
        if len(oob_indices) == 0:
            return

        # 获取当前树模型
        tree = self.estimators_[tree_idx]
        # 提取袋外样本的特征子集
        X_oob = X[oob_indices][:, feature_indices]
        # 预测袋外样本的概率分布
        y_oob_pred_proba = tree.predict_proba(X_oob)

        # 累加预测结果到全局估计器
        for local_idx, global_idx in enumerate(oob_indices):
            # 累加概率预测
            oob_pred_sum[global_idx] += y_oob_pred_proba[local_idx]
            # 增加预测计数
            oob_count[global_idx] += 1

    def _compute_feature_importances(self, n_features):
        """计算特征重要性（基于基尼不纯度减少）"""
        importances = np.zeros(n_features)  # 初始化特征重要性数组

        # 遍历所有树及其使用的特征
        for tree, feature_indices in zip(self.estimators_, self.estimator_features_):
            # 获取当前树的特征重要性
            tree_importances = tree.feature_importances_

            # 将重要性值累加到原始特征位置
            for tree_feat_idx, feat_idx in enumerate(feature_indices):
                importances[feat_idx] += tree_importances[tree_feat_idx]

        # 归一化重要性值（使总和为1）
        total_importance = importances.sum()
        if total_importance > 0:
            importances /= total_importance

        return importances  # 返回归一化后的重要性

    def _finalize_fit(self, X, y, oob_pred_sum=None, oob_count=None):
        """完成训练过程并设置模型状态"""
        self.is_fitted = True  # 标记模型已训练
        self.classes_ = np.unique(y)  # 获取唯一类别标签
        self.n_classes_ = len(self.classes_)  # 类别数量
        self.n_features_ = X.shape[1]  # 特征数量

        # 计算袋外分数（如果启用）
        if self.oob_score and oob_pred_sum is not None and oob_count is not None:
            # 创建有效OOB样本掩码（至少被预测一次）
            valid_oob_mask = oob_count > 0
            num_valid = np.sum(valid_oob_mask)  # 有效样本数

            if num_valid > 0:
                # 计算平均概率预测 [有效样本数, 类别数]
                oob_avg_proba = oob_pred_sum[valid_oob_mask] / oob_count[valid_oob_mask, np.newaxis]
                # 获取预测类别（最大概率对应类别）
                y_oob_pred = np.argmax(oob_avg_proba, axis=1)
                # 获取真实标签
                y_oob_true = y[valid_oob_mask]
                # 计算准确率作为OOB分数
                self.oob_score_ = accuracy_score(y_oob_true, y_oob_pred)

                # 输出调试信息
                if self.verbose > 0:
                    print(f"OOB分数: {self.oob_score_:.4f} (基于{num_valid}个样本)")
            else:
                # 无有效OOB样本时发出警告
                warnings.warn("没有有效的OOB样本，无法计算OOB分数")
                self.oob_score_ = None
        else:
            self.oob_score_ = None  # 未启用OOB评分
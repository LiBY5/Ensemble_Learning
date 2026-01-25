"""测试随机森林实现 - tests/test_random_forest.py"""

import numpy as np  # 导入NumPy库，用于数值计算
import sys  # 导入系统模块，用于修改Python路径
import os  # 导入操作系统模块，用于文件路径操作

# 添加项目根目录到Python路径，以便导入自定义模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 从scikit-learn导入所需模块
from sklearn.datasets import load_breast_cancer, make_classification  # 导入数据集加载函数
from sklearn.model_selection import train_test_split  # 导入数据集划分函数
from sklearn.metrics import accuracy_score  # 导入准确率计算函数
from sklearn.ensemble import RandomForestClassifier as SklearnRF  # 导入sklearn的随机森林，起别名避免命名冲突

# 导入自定义的随机森林实现
from models.random_forest import RandomForestClassifier


def test_random_forest_basic():
    """测试随机森林基本功能"""
    print("=" * 60)  # 打印60个等号作为分隔线
    print("测试1: 随机森林基本功能")  # 打印测试标题
    print("=" * 60)  # 打印60个等号作为分隔线

    # 使用乳腺癌数据集（经典分类数据集）
    data = load_breast_cancer()  # 加载乳腺癌数据集
    X, y = data.data, data.target  # 提取特征数据X和标签数据y

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42  # 20%作为测试集，设置随机种子保证可重复性
    )

    # 创建我们的随机森林模型实例
    our_rf = RandomForestClassifier(
        n_estimators=50,  # 设置50棵树（较少以便快速测试）
        max_depth=5,  # 限制最大深度为5，防止过拟合
        max_features='sqrt',  # 每棵树使用sqrt(特征数)个特征
        oob_score=True,  # 启用袋外分数计算
        random_state=42,  # 设置随机种子保证可重复性
        verbose=1  # 设置详细输出级别为1（显示训练进度）
    )

    # 训练模型
    print("训练我们的随机森林...")  # 打印训练开始信息
    our_rf.fit(X_train, y_train)  # 在训练集上训练模型

    # 使用训练好的模型进行预测
    y_pred = our_rf.predict(X_test)  # 在测试集上进行预测
    our_acc = accuracy_score(y_test, y_pred)  # 计算预测准确率

    # 打印结果
    print(f"\n我们的随机森林结果:")  # 打印结果标题
    print(f"  测试准确率: {our_acc:.4f}")  # 打印准确率，保留4位小数
    if our_rf.oob_score_ is not None:  # 如果计算了袋外分数
        print(f"  OOB分数: {our_rf.oob_score_:.4f}")  # 打印袋外分数

    # 特征重要性分析
    importances = our_rf.feature_importances_  # 获取特征重要性数组
    print(f"  特征重要性形状: {importances.shape}")  # 打印重要性数组的形状
    # 找到最重要的5个特征（按重要性降序排列）
    print(f"  最重要特征索引: {np.argsort(importances)[-5:][::-1]}")

    return our_rf, our_acc  # 返回模型和准确率供后续使用


def test_random_forest_vs_sklearn():
    """与sklearn的随机森林对比"""
    print("\n" + "=" * 60)  # 打印换行和60个等号作为分隔线
    print("测试2: 与sklearn的随机森林对比")  # 打印测试标题
    print("=" * 60)  # 打印60个等号作为分隔线

    # 生成模拟分类数据（更适合对比测试）
    X, y = make_classification(
        n_samples=1000,  # 生成1000个样本
        n_features=30,  # 30个特征
        n_informative=20,  # 其中20个是信息特征
        n_redundant=5,  # 5个是冗余特征
        n_classes=3,  # 3个类别（多分类问题）
        random_state=42  # 设置随机种子保证可重复性
    )

    # 划分训练集和测试集（分层抽样保持类别比例）
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y  # 30%作为测试集，分层抽样
    )

    # 创建我们的随机森林实现
    our_rf = RandomForestClassifier(
        n_estimators=100,  # 100棵树
        max_depth=10,  # 最大深度10
        max_features='sqrt',  # 使用sqrt(特征数)个特征
        oob_score=True,  # 启用袋外分数
        random_state=42,  # 随机种子
        verbose=0  # 不显示详细输出
    )

    # 创建sklearn的随机森林实现
    sklearn_rf = SklearnRF(
        n_estimators=100,  # 100棵树
        max_depth=10,  # 最大深度10
        max_features='sqrt',  # 使用sqrt(特征数)个特征
        oob_score=True,  # 启用袋外分数
        random_state=42,  # 随机种子
        n_jobs=-1  # 使用所有CPU核心并行计算
    )

    # 训练我们的实现
    print("训练我们的随机森林...")  # 打印训练开始信息
    our_rf.fit(X_train, y_train)  # 训练我们的模型
    our_pred = our_rf.predict(X_test)  # 在测试集上预测
    our_acc = accuracy_score(y_test, our_pred)  # 计算准确率

    # 训练sklearn的实现
    print("训练sklearn的随机森林...")  # 打印训练开始信息
    sklearn_rf.fit(X_train, y_train)  # 训练sklearn模型
    sklearn_pred = sklearn_rf.predict(X_test)  # 在测试集上预测
    sklearn_acc = accuracy_score(y_test, sklearn_pred)  # 计算准确率

    # 对比结果（格式化输出）
    print(f"\n对比结果:")  # 打印对比标题
    # 打印表头
    print(f"{'':<25} {'我们的实现':<12} {'sklearn':<12} {'差异':<10}")
    print("-" * 60)  # 打印分隔线
    # 打印准确率对比
    print(f"{'测试准确率':<25} {our_acc:.4f}       {sklearn_acc:.4f}       {abs(our_acc - sklearn_acc):.4f}")
    # 打印OOB分数对比
    print(
        f"{'OOB分数':<25} {our_rf.oob_score_:.4f}       {sklearn_rf.oob_score_:.4f}       {abs(our_rf.oob_score_ - sklearn_rf.oob_score_):.4f}")

    # 特征重要性对比
    our_importances = our_rf.feature_importances_  # 获取我们的特征重要性
    sklearn_importances = sklearn_rf.feature_importances_  # 获取sklearn的特征重要性

    # 计算两个重要性数组的皮尔逊相关系数
    correlation = np.corrcoef(our_importances, sklearn_importances)[0, 1]
    print(f"\n特征重要性相关性: {correlation:.4f}")  # 打印相关性系数

    return our_acc, sklearn_acc, correlation  # 返回准确率和相关性供后续分析


def test_random_forest_parameters():
    """测试随机森林不同参数配置 - 修复参数传递问题"""
    print("\n" + "=" * 60)  # 打印换行和60个等号作为分隔线
    print("测试3: 随机森林不同参数配置")  # 打印测试标题
    print("=" * 60)  # 打印60个等号作为分隔线

    # 生成新的模拟数据用于参数测试
    X, y = make_classification(n_samples=500, n_features=20, random_state=42)

    # 定义要测试的不同参数配置
    configurations = [
        ("默认配置", {"n_estimators": 50}),  # 只设置树的数量为50
        ("深度限制", {"n_estimators": 50, "max_depth": 3}),  # 添加深度限制
        ("更多特征", {"n_estimators": 50, "max_features": 0.8}),  # 使用80%的特征
        ("更少特征", {"n_estimators": 50, "max_features": 0.3}),  # 使用30%的特征
        ("更多树", {"n_estimators": 200}),  # 使用更多树（200棵）
        ("无自助采样", {"n_estimators": 50, "bootstrap": False}),  # 不使用bootstrap采样
    ]

    # 遍历每种配置进行测试
    for name, params in configurations:
        print(f"\n配置: {name}")  # 打印当前配置名称
        # 创建随机森林实例，传入当前配置参数
        rf = RandomForestClassifier(
            random_state=42,  # 固定随机种子
            verbose=0,  # 不显示详细输出
            **params  # 解包参数字典
        )

        rf.fit(X, y)  # 在整个数据集上训练（这里为了测试简单，没有划分训练测试集）
        acc = rf.score(X, y)  # 计算在训练集上的准确率（注意：这不是泛化性能的可靠指标）

        # 获取特征重要性信息
        if rf.feature_importances_ is not None:  # 如果特征重要性已计算
            top_feature = np.argmax(rf.feature_importances_)  # 找到最重要的特征索引
            importance_val = rf.feature_importances_[top_feature]  # 获取该特征的重要性值
        else:  # 如果特征重要性未计算
            top_feature = -1  # 设置为-1表示未知
            importance_val = 0  # 重要性值为0

        # 打印当前配置的结果
        print(f"  准确率={acc:.4f}, 最重要特征={top_feature}({importance_val:.3f})")


if __name__ == "__main__":
    """主程序入口：执行所有测试"""
    print("开始测试随机森林实现")  # 打印开始信息
    print("=" * 60)  # 打印分隔线

    # 测试1: 基本功能测试
    model, acc1 = test_random_forest_basic()

    # 测试2: 与sklearn对比测试
    acc2, acc_sklearn, corr = test_random_forest_vs_sklearn()

    # 测试3: 参数配置测试
    test_random_forest_parameters()

    print("\n" + "=" * 60)  # 打印换行和分隔线
    print("所有测试完成!")  # 打印完成信息
    print("=" * 60)  # 打印分隔线
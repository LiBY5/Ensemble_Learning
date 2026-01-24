"""测试Bagging实现 - 验证自定义Bagging分类器的正确性和性能"""
import numpy as np  # 导入NumPy库，用于数值计算
import sys  # 导入sys模块，用于操作系统相关功能
import os  # 导入os模块，用于操作系统相关功能

# 将项目根目录添加到系统路径，以便能够导入自定义模块
# sys.path.append用于添加模块搜索路径
# os.path.dirname用于获取父目录路径
# os.path.abspath用于获取绝对路径
# 这行代码将项目根目录添加到Python路径，使得可以从models目录导入模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入sklearn相关模块
from sklearn.datasets import make_classification  # 用于生成模拟分类数据集
from sklearn.tree import DecisionTreeClassifier  # 决策树分类器，作为基学习器
from sklearn.model_selection import train_test_split  # 用于数据集分割
from sklearn.metrics import accuracy_score  # 用于计算分类准确率
from sklearn.ensemble import BaggingClassifier as SklearnBagging  # sklearn的Bagging实现，用于对比

# 导入自定义的Bagging分类器
from models.bagging import BaggingClassifier


def test_bagging_basic():
    """测试Bagging基本功能 - 验证自定义Bagging分类器的基本功能是否正常工作"""
    print("=" * 60)  # 打印分隔线，美化输出格式
    print("测试1: Bagging基本功能")  # 打印测试标题
    print("=" * 60)  # 打印分隔线

    # 生成模拟分类数据集
    # make_classification函数参数说明：
    # n_samples=500: 生成500个样本
    # n_features=20: 每个样本有20个特征
    # n_informative=15: 其中15个是有效特征（与标签相关）
    # n_redundant=5: 5个是冗余特征（由有效特征线性组合生成）
    # n_classes=2: 二分类问题
    # random_state=42: 随机种子，确保结果可重现
    X, y = make_classification(
        n_samples=500,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_classes=2,
        random_state=42
    )

    # 将数据集分割为训练集和测试集
    # train_test_split函数参数说明：
    # test_size=0.2: 测试集占20%，训练集占80%
    # random_state=42: 随机种子，确保分割可重现
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 创建基学习器：决策树分类器
    # DecisionTreeClassifier参数说明：
    # max_depth=5: 决策树最大深度为5，防止过拟合
    # random_state=42: 随机种子，确保决策树构建可重现
    base_tree = DecisionTreeClassifier(max_depth=5, random_state=42)

    # 创建我们的Bagging分类器实例
    # BaggingClassifier参数说明：
    # base_estimator=base_tree: 使用决策树作为基学习器
    # n_estimators=10: 集成10个基学习器
    # max_samples=0.8: 每个基学习器使用80%的样本
    # max_features=0.8: 每个基学习器使用80%的特征
    # bootstrap=True: 使用有放回采样
    # oob_score=True: 计算袋外（OOB）分数
    # random_state=42: 随机种子，确保结果可重现
    # verbose=1: 显示训练进度信息
    our_bagging = BaggingClassifier(
        base_estimator=base_tree,
        n_estimators=10,
        max_samples=0.8,
        max_features=0.8,
        bootstrap=True,
        oob_score=True,
        random_state=42,
        verbose=1
    )

    # 训练自定义Bagging分类器
    print("训练我们的Bagging分类器...")
    our_bagging.fit(X_train, y_train)  # 调用fit方法进行训练

    # 使用训练好的模型进行预测
    y_pred = our_bagging.predict(X_test)  # 对测试集进行预测
    our_acc = accuracy_score(y_test, y_pred)  # 计算预测准确率

    # 打印结果信息
    print(f"\n我们的Bagging结果:")
    print(f"  训练集大小: {len(X_train)}")  # 打印训练集样本数
    print(f"  测试集大小: {len(X_test)}")  # 打印测试集样本数
    print(f"  基学习器数量: {len(our_bagging.estimators_)}")  # 打印实际训练的学习器数量
    print(f"  测试准确率: {our_acc:.4f}")  # 打印测试准确率，保留4位小数
    if our_bagging.oob_score_ is not None:  # 如果计算了OOB分数
        print(f"  OOB分数: {our_bagging.oob_score_:.4f}")  # 打印OOB分数

    # 验证模型属性，确保模型正确训练
    print(f"\n验证模型属性:")
    print(f"  is_fitted: {our_bagging.is_fitted}")  # 检查是否已训练
    print(f"  n_classes_: {our_bagging.n_classes_}")  # 打印类别数量
    print(f"  n_features_: {our_bagging.n_features_}")  # 打印特征数量
    print(f"  classes_: {our_bagging.classes_}")  # 打印所有类别标签

    # 返回训练好的模型和准确率，供后续使用
    return our_bagging, our_acc


def test_bagging_vs_sklearn():
    """与sklearn的Bagging对比 - 验证自定义实现与sklearn官方实现的一致性"""
    print("\n" + "=" * 60)  # 打印空行和分隔线
    print("测试2: 与sklearn的Bagging对比")  # 打印测试标题
    print("=" * 60)  # 打印分隔线

    # 生成更复杂的模拟数据集（三分类问题）
    X, y = make_classification(
        n_samples=1000,  # 1000个样本
        n_features=20,  # 20个特征
        n_informative=15,  # 15个有效特征
        n_redundant=5,  # 5个冗余特征
        n_classes=3,  # 三分类问题，更复杂
        random_state=42  # 随机种子
    )

    # 分层分割数据集，确保各类别在训练集和测试集中的分布一致
    # stratify=y: 根据y的类别分布进行分层采样
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # 创建我们的Bagging分类器实例
    our_bagging = BaggingClassifier(
        base_estimator=DecisionTreeClassifier(max_depth=5, random_state=42),
        n_estimators=20,  # 20个基学习器
        max_samples=0.7,  # 每个学习器使用70%样本
        max_features=0.7,  # 每个学习器使用70%特征
        bootstrap=True,  # 有放回采样
        oob_score=True,  # 计算OOB分数
        random_state=42,  # 随机种子
        verbose=0  # 不显示训练进度
    )

    # 创建sklearn的Bagging分类器实例
    # 注意：sklearn中参数名为estimator而不是base_estimator
    sklearn_bagging = SklearnBagging(
        estimator=DecisionTreeClassifier(max_depth=5, random_state=42),
        n_estimators=20,
        max_samples=0.7,
        max_features=0.7,
        bootstrap=True,
        oob_score=True,
        random_state=42
    )

    # 训练我们的实现
    print("训练我们的Bagging实现...")
    our_bagging.fit(X_train, y_train)  # 训练自定义Bagging
    our_pred = our_bagging.predict(X_test)  # 对测试集进行预测
    our_acc = accuracy_score(y_test, our_pred)  # 计算准确率

    # 训练sklearn的实现
    print("训练sklearn的Bagging...")
    sklearn_bagging.fit(X_train, y_train)  # 训练sklearn Bagging
    sklearn_pred = sklearn_bagging.predict(X_test)  # 对测试集进行预测
    sklearn_acc = accuracy_score(y_test, sklearn_pred)  # 计算准确率

    # 对比两种实现的结果
    print(f"\n对比结果:")
    # 打印表头，使用格式化字符串确保对齐
    print(f"{'':<20} {'我们的实现':<12} {'sklearn':<12} {'差异':<10}")
    print("-" * 60)  # 打印分隔线
    # 打印测试准确率对比
    print(f"{'测试准确率':<20} {our_acc:.4f}       {sklearn_acc:.4f}       {abs(our_acc - sklearn_acc):.4f}")
    # 打印OOB分数对比
    print(f"{'OOB分数':<20} {our_bagging.oob_score_:.4f}       {sklearn_bagging.oob_score_:.4f}       {abs(our_bagging.oob_score_ - sklearn_bagging.oob_score_):.4f}")

    # 检查概率预测功能
    print(f"\n检查概率预测:")
    # 获取前5个测试样本的概率预测
    our_proba = our_bagging.predict_proba(X_test[:5])
    sklearn_proba = sklearn_bagging.predict_proba(X_test[:5])
    print(f"前5个样本的概率预测形状: {our_proba.shape}")  # 打印概率矩阵形状
    # 检查概率和是否为1（概率预测的合法性检查）
    print(f"概率和是否为1: {np.allclose(our_proba.sum(axis=1), 1.0)}")

    # 返回两种实现的准确率，供后续分析
    return our_acc, sklearn_acc


def test_bagging_variations():
    """测试Bagging的不同变体 - 验证不同参数配置下的模型表现"""
    print("\n" + "=" * 60)  # 打印空行和分隔线
    print("测试3: Bagging不同变体")  # 打印测试标题
    print("=" * 60)  # 打印分隔线

    # 生成较小的模拟数据集，用于快速测试不同变体
    X, y = make_classification(n_samples=200, n_features=10, random_state=42)

    # 定义要测试的不同变体配置
    # 每个变体是一个元组：(变体名称, 参数字典)
    variations = [
        ("有放回采样", {"bootstrap": True, "bootstrap_features": False}),  # 标准Bagging
        ("无放回采样", {"bootstrap": False, "bootstrap_features": False}),  # Pasting
        ("特征有放回", {"bootstrap": True, "bootstrap_features": True}),  # 特征有放回采样
        ("小样本比例", {"bootstrap": True, "max_samples": 0.5}),  # 使用较少样本
        ("小特征比例", {"bootstrap": True, "max_features": 0.5}),  # 使用较少特征
    ]

    # 创建基学习器
    base_tree = DecisionTreeClassifier(max_depth=3, random_state=42)

    # 遍历所有变体配置进行测试
    for name, params in variations:
        # 使用当前参数配置创建Bagging分类器
        bagging = BaggingClassifier(
            base_estimator=base_tree,  # 使用决策树作为基学习器
            n_estimators=5,  # 只使用5个基学习器，加快测试速度
            random_state=42,  # 随机种子
            **params  # 展开参数字典，传入变体特定参数
        )

        # 训练模型（在整个数据集上训练和测试，仅用于演示）
        bagging.fit(X, y)
        # 计算在整个数据集上的准确率（注意：这不是严格的评估方式，仅用于演示）
        acc = bagging.score(X, y)

        # 打印当前变体的结果
        # 从参数中获取bootstrap_features值，判断特征采样方式
        feature_sampling = '有放回' if params.get('bootstrap_features', False) else '无放回'
        print(f"{name:15s}: 准确率={acc:.4f}, 特征采样方式={feature_sampling}")


if __name__ == "__main__":
    """主函数 - 执行所有测试"""
    print("开始测试Bagging实现")  # 打印开始信息
    print("=" * 60)  # 打印分隔线

    # 测试1: 基本功能
    # 调用test_bagging_basic函数，获取训练好的模型和准确率
    model, acc1 = test_bagging_basic()

    # 测试2: 与sklearn对比
    # 调用test_bagging_vs_sklearn函数，获取两种实现的准确率
    acc2, acc_sklearn = test_bagging_vs_sklearn()

    # 测试3: 不同变体
    # 调用test_bagging_variations函数，测试不同参数配置
    test_bagging_variations()

    # 打印测试完成信息
    print("\n" + "=" * 60)
    print("所有测试完成!")
    print("=" * 60)
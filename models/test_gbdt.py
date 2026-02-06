"""测试GBDT实现"""  # 模块文档字符串，说明本模块的作用

import numpy as np  # 导入numpy库，用于数值计算
import matplotlib.pyplot as plt  # 导入matplotlib的pyplot模块，用于数据可视化
from sklearn.datasets import make_classification, make_regression  # 从sklearn导入数据生成函数
from sklearn.model_selection import train_test_split  # 从sklearn导入数据集划分函数
from sklearn.ensemble import GradientBoostingClassifier as SklearnGBC  # 导入sklearn的GBDT分类器，并重命名
from sklearn.ensemble import GradientBoostingRegressor as SklearnGBR  # 导入sklearn的GBDT回归器，并重命名
from sklearn.metrics import accuracy_score, mean_squared_error, roc_auc_score  # 导入评估指标函数

# 从自定义模块导入我们实现的GBDT模型
from models.boosting import (GradientBoostingRegressor,
                            GradientBoostingClassifier)

def test_gbdt_regression():
    """测试GBDT回归器"""  # 函数文档字符串，说明函数功能

    # 生成回归数据
    # make_regression函数生成适用于回归问题的模拟数据
    # n_samples=1000: 生成1000个样本
    # n_features=10: 每个样本有10个特征
    # n_informative=8: 其中8个特征是有信息量的（与目标变量相关）
    # noise=20: 添加20的高斯噪声
    # random_state=42: 随机种子，保证结果可重现
    X, y = make_regression(
        n_samples=1000, n_features=10, n_informative=8,
        noise=20, random_state=42
    )

    # 划分训练集和测试集
    # test_size=0.2: 20%的数据作为测试集
    # random_state=42: 随机种子，保证划分可重现
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 初始化我们实现的GBDT回归器
    our_gbdt = GradientBoostingRegressor(
        loss='ls',  # 损失函数：最小二乘损失
        learning_rate=0.1,  # 学习率（步长）
        n_estimators=100,  # 弱学习器（树）的数量
        max_depth=3,  # 每棵树的最大深度
        subsample=0.8,  # 子采样比例（每棵树使用80%的样本）
        random_state=42,  # 随机种子
        verbose=0  # 训练时不输出详细信息
    )

    # 初始化sklearn的GBDT回归器
    sklearn_gbdt = SklearnGBR(
        loss='squared_error',  # 损失函数：平方误差（相当于我们的'ls'）
        learning_rate=0.1,  # 学习率
        n_estimators=100,  # 树的数量
        max_depth=3,  # 树的最大深度
        subsample=0.8,  # 子采样比例
        random_state=42  # 随机种子
    )

    print("训练我们的GBDT回归器...")  # 打印提示信息
    our_gbdt.fit(X_train, y_train)  # 训练我们实现的模型
    our_pred = our_gbdt.predict(X_test)  # 对测试集进行预测
    our_mse = mean_squared_error(y_test, our_pred)  # 计算均方误差

    print(f"\n我们的实现:")  # 打印结果标题
    print(f"  测试MSE: {our_mse:.4f}")  # 打印测试MSE，保留4位小数
    # 打印训练损失历史的前5个值
    print(f"  训练损失历史: {our_gbdt.train_score_[:5]}...")

    print("\n训练sklearn的GBDT回归器...")  # 打印提示信息
    sklearn_gbdt.fit(X_train, y_train)  # 训练sklearn模型
    sklearn_pred = sklearn_gbdt.predict(X_test)  # 对测试集进行预测
    sklearn_mse = mean_squared_error(y_test, sklearn_pred)  # 计算均方误差

    print(f"\nsklearn实现:")  # 打印结果标题
    print(f"  测试MSE: {sklearn_mse:.4f}")  # 打印测试MSE

    # 可视化训练过程：调用自定义函数绘制训练损失曲线
    visualize_training_curve(our_gbdt.train_score_, sklearn_gbdt.train_score_)

    return our_mse, sklearn_mse  # 返回两个模型的MSE

def test_gbdt_classification():
    """测试GBDT分类器"""  # 函数文档字符串

    # 生成二分类数据
    # make_classification函数生成适用于分类问题的模拟数据
    # n_samples=1000: 1000个样本
    # n_features=20: 20个特征
    # n_informative=15: 15个有信息量的特征
    # n_classes=2: 二分类问题
    # random_state=42: 随机种子
    X, y = make_classification(
        n_samples=1000, n_features=20, n_informative=15,
        n_classes=2, random_state=42
    )

    # 划分训练集和测试集，保持类别分布一致
    # stratify=y: 按y的类别分布进行分层抽样
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 初始化我们实现的GBDT分类器
    our_gbdt = GradientBoostingClassifier(
        loss='deviance',  # 损失函数：偏差（对数似然）
        learning_rate=0.1,  # 学习率
        n_estimators=100,  # 树的数量
        max_depth=3,  # 树的最大深度
        subsample=0.8,  # 子采样比例
        random_state=42,  # 随机种子
        verbose=0  # 训练时不输出详细信息
    )

    # 初始化sklearn的GBDT分类器
    sklearn_gbdt = SklearnGBC(
        loss='log_loss',  # 损失函数：对数损失（相当于我们的'deviance'）
        learning_rate=0.1,  # 学习率
        n_estimators=100,  # 树的数量
        max_depth=3,  # 树的最大深度
        subsample=0.8,  # 子采样比例
        random_state=42  # 随机种子
    )

    print("\n训练我们的GBDT分类器...")  # 打印提示信息
    our_gbdt.fit(X_train, y_train)  # 训练我们实现的模型
    our_pred = our_gbdt.predict(X_test)  # 对测试集进行类别预测
    our_proba = our_gbdt.predict_proba(X_test)[:, 1]  # 获取正类的概率预测
    our_acc = accuracy_score(y_test, our_pred)  # 计算准确率
    our_auc = roc_auc_score(y_test, our_proba)  # 计算AUC值

    print(f"\n我们的实现:")  # 打印结果标题
    print(f"  测试准确率: {our_acc:.4f}")  # 打印准确率
    print(f"  AUC: {our_auc:.4f}")  # 打印AUC值

    print("\n训练sklearn的GBDT分类器...")  # 打印提示信息
    sklearn_gbdt.fit(X_train, y_train)  # 训练sklearn模型
    sklearn_pred = sklearn_gbdt.predict(X_test)  # 对测试集进行类别预测
    sklearn_proba = sklearn_gbdt.predict_proba(X_test)[:, 1]  # 获取正类的概率预测
    sklearn_acc = accuracy_score(y_test, sklearn_pred)  # 计算准确率
    sklearn_auc = roc_auc_score(y_test, sklearn_proba)  # 计算AUC值

    print(f"\nsklearn实现:")  # 打印结果标题
    print(f"  测试准确率: {sklearn_acc:.4f}")  # 打印准确率
    print(f"  AUC: {sklearn_auc:.4f}")  # 打印AUC值

    # 可视化特征重要性：调用自定义函数
    visualize_feature_importance(our_gbdt, sklearn_gbdt, X_train.shape[1])

    return our_acc, sklearn_acc  # 返回两个模型的准确率

def visualize_training_curve(our_scores, sklearn_scores):
    """可视化训练曲线"""  # 函数文档字符串

    plt.figure(figsize=(10, 6))  # 创建图形，设置大小为10×6英寸

    # 绘制我们实现的模型的训练损失曲线
    # range(1, len(our_scores) + 1): x轴坐标，从1到损失值个数
    # our_scores: y轴坐标，训练损失值
    # 'b-': 蓝色实线
    # linewidth=2: 线宽为2
    plt.plot(range(1, len(our_scores) + 1), our_scores,
             'b-', label='我们的实现', linewidth=2)

    # 绘制sklearn模型的训练损失曲线
    # alpha=0.7: 设置透明度为0.7
    plt.plot(range(1, len(sklearn_scores) + 1), sklearn_scores,
             'r-', label='sklearn实现', linewidth=2, alpha=0.7)

    plt.xlabel('迭代次数')  # 设置x轴标签
    plt.ylabel('训练损失')  # 设置y轴标签
    plt.title('GBDT训练损失曲线')  # 设置图形标题
    plt.legend()  # 显示图例
    plt.grid(True, alpha=0.3)  # 显示网格，透明度0.3
    plt.yscale('log')  # 设置y轴为对数刻度

    plt.tight_layout()  # 自动调整子图参数，使之填充整个图像区域

    # 设置字体为系统自带的中文字体
    # 解决matplotlib默认字体不支持中文的问题
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']

    # 解决负号显示问题
    # 设置axes.unicode_minus参数为False，解决负号显示为方块的问题
    plt.rcParams['axes.unicode_minus'] = False

    # 保存图形到指定路径
    # dpi=150: 设置分辨率为150点每英寸
    plt.savefig('../results/figures/day3_gbdt_training_curve.png', dpi=150)
    plt.show()  # 显示图形

def visualize_feature_importance(our_model, sklearn_model, n_features):
    """可视化特征重要性"""  # 函数文档字符串

    # 注意：我们的实现没有计算特征重要性，这里用sklearn的
    # 检查sklearn模型是否有feature_importances_属性
    if hasattr(sklearn_model, 'feature_importances_'):
        importances = sklearn_model.feature_importances_  # 获取特征重要性
        indices = np.argsort(importances)[::-1][:15]  # 按重要性降序排序，取前15个索引

        plt.figure(figsize=(12, 6))  # 创建图形，设置大小（适当加宽以便显示标签）
        bars = plt.bar(range(len(indices)), importances[indices])  # 绘制条形图，保存bars对象
        plt.xticks(range(len(indices)), indices, rotation=45)  # 设置x轴刻度标签，旋转45度
        plt.xlabel('特征索引')  # 设置x轴标签
        plt.ylabel('重要性')  # 设置y轴标签
        plt.title('GBDT特征重要性 (Top 15)')  # 设置图形标题
        plt.grid(True, alpha=0.3, axis='y')  # 显示y轴方向的网格

        # 为每个条形添加数值标签
        for i, (bar, importance) in enumerate(zip(bars, importances[indices])):
            height = bar.get_height()  # 获取条形的高度（重要性数值）

            # 设置标签文本：保留4位小数
            label_text = f'{importance:.4f}'

            # 确定标签位置：在条形顶部上方一点
            # 如果高度很小，标签放在条形内部顶部；否则放在外部
            if height < 0.01:  # 如果重要性值很小
                # 在条形内部顶部显示，颜色为白色以提高可读性
                plt.text(bar.get_x() + bar.get_width()/2., height,
                         label_text, ha='center', va='bottom',
                         color='white', fontweight='bold', fontsize=9)
            else:
                # 在条形顶部上方显示
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.003,
                         label_text, ha='center', va='bottom',
                         fontsize=9)

        plt.tight_layout()  # 自动调整子图参数

        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']

        # 解决负号显示问题
        plt.rcParams['axes.unicode_minus'] = False

        # 保存图形
        plt.savefig('../results/figures/day3_gbdt_feature_importance.png', dpi=150, bbox_inches='tight')
        plt.show()  # 显示图形

# 主程序入口
if __name__ == "__main__":
    import os  # 导入os模块，用于操作系统功能

    # 创建结果目录（如果不存在）
    # exist_ok=True: 如果目录已存在，不会抛出异常
    os.makedirs('../results/figures', exist_ok=True)

    # 测试回归模型
    our_mse, sklearn_mse = test_gbdt_regression()
    print(f"\n回归MSE差异: {abs(our_mse - sklearn_mse):.4f}")

    # 测试分类模型
    our_acc, sklearn_acc = test_gbdt_classification()
    print(f"\n分类准确率差异: {abs(our_acc - sklearn_acc):.4f}")
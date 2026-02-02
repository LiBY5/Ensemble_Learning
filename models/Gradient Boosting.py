import numpy as np
def gradient_boosting_derivation():
    """梯度提升算法推导"""


    print("=" * 60)
    print("梯度提升算法数学推导")
    print("=" * 60)

    # 1. 问题定义
    print("\n1. 问题定义:")
    print("   目标：最小化损失函数 L(y, F(x))")
    print("   其中 F(x) 是我们的预测模型")

    # 2. 前向分步算法
    print("\n2. 前向分步算法（加法模型）:")
    print("   F(x) = F_0(x) + Σ_{t=1}^T ν * h_t(x)")
    print("   每步添加一个基学习器，逐步改进模型")

    # 3. 梯度下降视角
    print("\n3. 函数空间梯度下降:")
    print("   传统参数梯度下降：θ_{t} = θ_{t-1} - η * ∇_θ L")
    print("   函数空间梯度下降：F_t(x) = F_{t-1}(x) - ν * ∇_F L")
    print("   其中 ∇_F L 是损失函数对模型F的梯度")

    # 4. 负梯度（伪残差）计算
    print("\n4. 负梯度（伪残差）计算:")
    print("   对于每个样本 i，计算：")
    print("   r_{ti} = -[∂L(y_i, F(x_i))/∂F(x_i)]|_{F=F_{t-1}}")

    # 5. 不同损失函数的梯度
    print("\n5. 常见损失函数的负梯度:")

    # 模拟一些数据
    np.random.seed(42)
    y_true = np.random.randn(5)
    y_pred = np.random.randn(5)

    # 平方损失
    print("   a. 平方损失 L(y, F) = (y-F)²/2")
    print(f"      真实值: {y_true}")
    print(f"      预测值: {y_pred}")
    neg_grad_square = y_true - y_pred
    print(f"      负梯度 = y - F = {neg_grad_square}")

    # 绝对损失
    print("\n   b. 绝对损失 L(y, F) = |y-F|")
    neg_grad_abs = np.sign(y_true - y_pred)
    print(f"      负梯度 = sign(y-F) = {neg_grad_abs}")

    # 对数损失（二分类）
    print("\n   c. 对数损失（二分类）")
    print("      L(y, F) = log(1 + exp(-2yF)), 其中 y ∈ {-1, 1}")
    print("      负梯度 = 2y / (1 + exp(2yF))")

    # 6. 算法步骤
    print("\n" + "=" * 60)
    print("完整算法步骤:")
    print("=" * 60)

    steps = [
        "1. 初始化模型: F_0(x) = argmin_γ Σ_i L(y_i, γ)",
        "2. For t = 1 to T:",
        "   a. 计算负梯度: r_i = -∂L/∂F|_{F=F_{t-1}}",
        "   b. 训练基学习器 h_t 拟合负梯度: h_t(x) ≈ r",
        "   c. 计算最优步长: γ_t = argmin_γ Σ_i L(y_i, F_{t-1}(x_i) + γ·h_t(x_i))",
        "   d. 更新模型: F_t(x) = F_{t-1}(x) + ν·γ_t·h_t(x)",
        "3. 输出最终模型 F_T(x)"
    ]

    for step in steps:
        print(step)

    # 可视化梯度下降过程
    visualize_gradient_descent()


def visualize_gradient_descent():
    """可视化梯度下降过程"""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    # 创建一个简单的二次函数作为损失函数
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x, y)
    Z = X ** 2 + Y ** 2  # 简单的二次损失

    # 模拟梯度下降路径
    path = [(4, 4)]
    learning_rate = 0.1
    for _ in range(20):
        grad_x = 2 * path[-1][0]
        grad_y = 2 * path[-1][1]
        new_x = path[-1][0] - learning_rate * grad_x
        new_y = path[-1][1] - learning_rate * grad_y
        path.append((new_x, new_y))

    path = np.array(path)

    # 绘制3D图
    fig = plt.figure(figsize=(14, 6))

    # 3D曲面图
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7)
    ax1.plot(path[:, 0], path[:, 1], path[:, 0] ** 2 + path[:, 1] ** 2,
             'r-', linewidth=3, marker='o', markersize=8)
    ax1.set_xlabel('参数 θ1')
    ax1.set_ylabel('参数 θ2')
    ax1.set_zlabel('损失 L(θ)')
    ax1.set_title('参数空间梯度下降')

    # 2D等高线图
    ax2 = fig.add_subplot(122)
    contour = ax2.contour(X, Y, Z, levels=20, cmap='viridis', alpha=0.7)
    ax2.plot(path[:, 0], path[:, 1], 'r-', linewidth=3, marker='o', markersize=8)
    ax2.set_xlabel('参数 θ1')
    ax2.set_ylabel('参数 θ2')
    ax2.set_title('梯度下降路径（等高线）')
    plt.colorbar(contour, ax=ax2)

    plt.tight_layout()
    # 设置字体为系统自带的中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'KaiTi', 'FangSong']  # 设置中文字体
    # 解决负号显示问题
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方块的问题
    plt.savefig('../results/figures/day3_gradient_descent_visualization.png', dpi=150)
    plt.show()

    print("\n可视化说明：")
    print("• 红色路径：梯度下降的轨迹，逐步走向损失最低点")
    print("• 类比GBDT：每个点代表一个模型，每一步添加一个基学习器")


# 运行推导
gradient_boosting_derivation()
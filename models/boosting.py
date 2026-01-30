import numpy as np
import sympy as sp

def adaboost_formula_derivation():
    """推导AdaBoost关键公式"""
    print("=== AdaBoost关键公式推导 ===\n")

    # 定义符号
    epsilon_t = sp.symbols('epsilon_t', positive=True)  # 错误率
    alpha_t = sp.symbols('alpha_t')  # 基学习器权重

    print("1. 指数损失函数：L = Σ_i exp(-y_i * H(x_i))")
    print("2. 对于第t轮，添加基学习器h_t，权重α_t")
    print("3. 要最小化的损失：L_t = Σ_i D_t(i) * exp(-α_t * y_i * h_t(x_i))")
    print("\n4. 分离正确分类和错误分类的样本：")
    print("   • 正确分类：y_i * h_t(x_i) = 1")
    print("   • 错误分类：y_i * h_t(x_i) = -1")

    print("\n5. 最小化L_t对α_t求导：")
    print("   L_t = (1-ε_t)*exp(-α_t) + ε_t*exp(α_t)")
    print("   对α_t求导：dL_t/dα_t = -(1-ε_t)*exp(-α_t) + ε_t*exp(α_t)")
    print("   令导数为0：-(1-ε_t)*exp(-α_t) + ε_t*exp(α_t) = 0")
    print("   => ε_t*exp(α_t) = (1-ε_t)*exp(-α_t)")
    print("   => exp(2α_t) = (1-ε_t)/ε_t")
    print("   => α_t = 0.5 * ln((1-ε_t)/ε_t)")

    # 验证公式
    print("\n=== 公式验证 ===")

    # 测试不同错误率对应的权重
    test_errors = [0.1, 0.3, 0.4, 0.45, 0.49]
    for eps in test_errors:
        alpha = 0.5 * np.log((1 - eps) / eps)
        print(f"当 ε_t = {eps:.3f} 时，α_t = {alpha:.4f}")

    # 可视化错误率与权重的关系
    import matplotlib.pyplot as plt

    epsilons = np.linspace(0.01, 0.49, 100)
    alphas = 0.5 * np.log((1 - epsilons) / epsilons)

    plt.figure(figsize=(10, 6))
    plt.plot(epsilons, alphas, 'b-', linewidth=3)
    plt.xlabel('错误率 ε_t', fontsize=12)
    plt.ylabel('基学习器权重 α_t', fontsize=12)
    plt.title('AdaBoost: 错误率与基学习器权重关系', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    plt.axvline(x=0.5, color='r', linestyle='--', alpha=0.5, label='ε_t=0.5 (α_t=0)')
    plt.legend()
    # 设置字体为系统自带的中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'KaiTi', 'FangSong']  # 设置中文字体
    # 解决负号显示问题
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方块的问题
    plt.tight_layout()
    plt.show()

    print("\n关键观察：")
    print("1. 当ε_t<0.5时，α_t>0，基学习器有正贡献")
    print("2. 当ε_t接近0.5时，α_t接近0，贡献很小")
    print("3. 当ε_t>0.5时，α_t<0，会破坏集成效果")
    print("4. 因此要求基学习器错误率必须小于0.5")

# 运行推导
adaboost_formula_derivation()


def weight_update_visualization():
    """可视化样本权重更新"""
    import matplotlib.pyplot as plt

    np.random.seed(42)
    n_samples = 50
    y_true = np.random.choice([-1, 1], size=n_samples)  # 真实标签

    # 模拟基学习器的预测（有一定错误率）
    error_rate = 0.3
    y_pred = y_true.copy()
    error_indices = np.random.choice(n_samples, size=int(n_samples * error_rate), replace=False)
    y_pred[error_indices] = -y_true[error_indices]  # 反转标签，模拟错误

    # 初始化权重
    weights = np.ones(n_samples) / n_samples

    # 计算基学习器权重
    error = np.sum(weights * (y_pred != y_true))
    alpha = 0.5 * np.log((1 - error) / error)

    print(f"初始情况：")
    print(f"• 样本数量：{n_samples}")
    print(f"• 错误样本数：{len(error_indices)}")
    print(f"• 加权错误率：{error:.4f}")
    print(f"• 基学习器权重α：{alpha:.4f}")

    # 更新权重
    new_weights = weights * np.exp(-alpha * y_true * y_pred)
    new_weights /= new_weights.sum()  # 归一化

    # 可视化
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 1. 权重变化
    axes[0].bar(range(n_samples), weights, alpha=0.7, label='原始权重', color='blue', width=0.8)
    axes[0].bar(range(n_samples), new_weights, alpha=0.7, label='更新后权重', color='red', width=0.4)
    axes[0].set_xlabel('样本索引', fontsize=10)
    axes[0].set_ylabel('权重', fontsize=10)
    axes[0].set_title('样本权重更新对比', fontsize=12)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 2. 错误样本权重变化
    error_mask = (y_pred != y_true)
    correct_mask = (y_pred == y_true)

    axes[1].scatter(range(error_mask.sum()), weights[error_mask],
                    label='错误样本原始权重', alpha=0.8, s=80, color='red')
    axes[1].scatter(range(error_mask.sum()), new_weights[error_mask],
                    label='错误样本新权重', alpha=0.8, s=80, color='darkred', marker='s')
    axes[1].set_xlabel('错误样本索引', fontsize=10)
    axes[1].set_ylabel('权重', fontsize=10)
    axes[1].set_title('错误样本权重变化（增大）', fontsize=12)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # 3. 正确样本权重变化
    axes[2].scatter(range(correct_mask.sum()), weights[correct_mask],
                    label='正确样本原始权重', alpha=0.8, s=80, color='green')
    axes[2].scatter(range(correct_mask.sum()), new_weights[correct_mask],
                    label='正确样本新权重', alpha=0.8, s=80, color='darkgreen', marker='s')
    axes[2].set_xlabel('正确样本索引', fontsize=10)
    axes[2].set_ylabel('权重', fontsize=10)
    axes[2].set_title('正确样本权重变化（减小）', fontsize=12)
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('day3_weight_update.png', dpi=150, bbox_inches='tight')
    plt.show()

    # 统计信息
    print(f"\n权重更新统计：")
    print(f"• 错误样本权重平均变化：{(new_weights[error_mask] / weights[error_mask]).mean():.2f}倍")
    print(f"• 正确样本权重平均变化：{(new_weights[correct_mask] / weights[correct_mask]).mean():.2f}倍")
    print(f"• 错误样本总权重占比变化：{new_weights[error_mask].sum():.2%} (原始: {weights[error_mask].sum():.2%})")

    # 数学验证
    print(f"\n数学验证：")
    print(f"• exp(-α) = exp(-{alpha:.4f}) = {np.exp(-alpha):.4f} (正确样本权重乘数)")
    print(f"• exp(α) = exp({alpha:.4f}) = {np.exp(alpha):.4f} (错误样本权重乘数)")


# 运行可视化
weight_update_visualization()
"""Domain-Adversarial Neural Network (DANN) 组件。

Ganin et al., "Domain-Adversarial Training of Neural Networks", JMLR 2016.

用于多源域映射任务：在 encoder 输出的 bottleneck 上施加域对抗约束，
迫使不同源域的中间表示不可分辨。
"""

import torch
import torch.nn as nn


class GradientReversal(torch.autograd.Function):
    """梯度反转层。

    前向传播：恒等映射。
    反向传播：梯度乘以 -lambda，使 upstream 网络向"欺骗分类器"的方向更新。
    """

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_, None


def grad_reverse(x, lambda_=1.0):
    return GradientReversal.apply(x, lambda_)


class DomainClassifier(nn.Module):
    """域分类器 MLP。

    对 bottleneck 做全局平均池化后，经过两层 MLP 预测源域标签。
    GradientReversal 在前向调用中自动介入。

    Args:
        in_dim: bottleneck 通道数（池化后的特征维度）。
        hidden_dim: 隐藏层维度，不宜过大以防分类器过强。
        num_domains: 源域数量。
    """

    def __init__(self, in_dim: int, hidden_dim: int = 256, num_domains: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_domains),
        )

    def forward(self, x, lambda_=1.0):
        """前向传播。

        Args:
            x: bottleneck 池化后的特征 [B, in_dim]。
            lambda_: GRL 的梯度反转系数。

        Returns:
            domain_logits: [B, num_domains] 未归一化的分类分数。
        """
        x = grad_reverse(x, lambda_)
        return self.net(x)


def grl_lambda_schedule(current_step: int, total_steps: int, gamma: float = 10.0) -> float:
    """计算 GRL 的渐进 lambda 值。

    使用 sigmoid 调度：训练初期 lambda≈0（encoder 不受对抗影响），
    后期 lambda≈1（全力消除源域差异）。

    Args:
        current_step: 当前全局步数。
        total_steps: 总步数 = epochs × batches_per_epoch。
        gamma: 控制过渡陡峭程度，默认 10.0。

    Returns:
        lambda 值，范围 (0, 1)。
    """
    p = current_step / max(total_steps, 1)
    return 2.0 / (1.0 + torch.exp(torch.tensor(-gamma * p)).item()) - 1.0

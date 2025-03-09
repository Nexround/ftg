import torch
import torch.nn as nn


class LoKILinear(nn.Module):
    def __init__(self, original_linear, active_outputs):
        super().__init__()
        self.out_features = original_linear.out_features
        self.active_pos = sorted(active_outputs)
        self.fixed_pos = [
            i for i in range(self.out_features) if i not in self.active_pos
        ]

        # 参数验证
        if not all(0 <= idx < self.out_features for idx in self.active_pos):
            raise ValueError(f"激活索引必须在[0, {self.out_features-1}]范围内")
        if len(self.active_pos) != len(set(self.active_pos)):
            raise ValueError("激活索引包含重复值")

        # 分割权重矩阵
        W = original_linear.weight.data
        self.active_part = nn.Linear(
            original_linear.in_features, len(self.active_pos), bias=False
        )
        self.fixed_part = nn.Linear(
            original_linear.in_features, len(self.fixed_pos), bias=False
        )

        # 初始化可训练部分
        self.active_part.weight.data = W[self.active_pos].clone()
        self.active_part.weight.requires_grad = True
        # 固定非激活部分
        self.fixed_part.weight.data = W[self.fixed_pos].clone()
        self.fixed_part.weight.requires_grad = False

        # 处理偏置项
        if original_linear.bias is not None:
            b = original_linear.bias.data
            self.active_bias = nn.Parameter(
                b[self.active_pos].clone(), requires_grad=True
            )
            self.fixed_bias = nn.Parameter(
                b[self.fixed_pos].clone(), requires_grad=False
            )
        else:
            self.register_parameter("active_bias", None)
            self.register_parameter("fixed_bias", None)

    def forward(self, x):
        # 分别计算激活部分和固定部分
        active_out = self.active_part(x)
        fixed_out = self.fixed_part(x)

        # 添加偏置
        if self.active_bias is not None:
            active_out += self.active_bias
            fixed_out += self.fixed_bias

        # 合并输出保持原始顺序
        combined = torch.zeros(
            (x.size(0), x.size(1), self.out_features), device=x.device, dtype=x.dtype
        )
        combined[:, :, self.active_pos] = active_out
        combined[:, :, self.fixed_pos] = fixed_out.detach()  # 阻断梯度传播

        return combined

    def get_trainable_params(self):
        """获取当前可训练参数的数量统计"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def replace_all_target_linear_qwen(model, active_neurons):
    for layer_idx in range(model.config.num_hidden_layers):
        target_linear = model.model.layers[layer_idx].mlp.down_proj
        new_linear = LoKILinear(target_linear, active_neurons[layer_idx])
        new_linear.to(next(target_linear.parameters()).device)
        setattr(model.model.layers[layer_idx].mlp, "down_proj", new_linear)
        print(f"成功替换层{layer_idx}")

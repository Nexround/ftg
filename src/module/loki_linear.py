import torch
import torch.nn as nn


class LoKILinear(nn.Module):
    def __init__(self, original_linear, target_neurons):
        super().__init__()
        self.out_features = original_linear.out_features
        self.active_pos = sorted(target_neurons)
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
        
def restore_original_linears(model):
    for layer_idx in range(model.config.num_hidden_layers):
        layer = model.model.layers[layer_idx]
        mlp = layer.mlp
        
        # 检查当前层的down_proj是否是LoKILinear实例
        if hasattr(mlp, 'down_proj') and isinstance(mlp.down_proj, LoKILinear):
            loki_layer = mlp.down_proj
            
            # 创建标准线性层
            original_linear = nn.Linear(
                in_features = loki_layer.active_part.in_features,
                out_features = loki_layer.out_features,
                bias = loki_layer.active_bias is not None
            )
            
            # 合并权重
            combined_weight = torch.zeros_like(original_linear.weight)
            combined_weight[loki_layer.active_pos] = loki_layer.active_part.weight.data
            combined_weight[loki_layer.fixed_pos] = loki_layer.fixed_part.weight.data
            
            # 合并偏置
            if loki_layer.active_bias is not None:
                combined_bias = torch.zeros_like(original_linear.bias)
                combined_bias[loki_layer.active_pos] = loki_layer.active_bias.data
                combined_bias[loki_layer.fixed_pos] = loki_layer.fixed_bias.data
                original_linear.bias.data = combined_bias
            
            # 设置权重并保持设备一致
            original_linear.weight.data = combined_weight
            original_linear = original_linear.to(loki_layer.active_part.weight.device)
            
            # 替换回原始结构
            setattr(mlp, 'down_proj', original_linear)
            print(f"成功还原层{layer_idx}的down_proj")
    
    return model
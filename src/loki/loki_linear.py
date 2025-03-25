import torch
import torch.nn as nn


class LoKILinear(nn.Module):
    def __init__(self, original_linear, target_neurons):
        super().__init__()
        self.out_features = original_linear.out_features
        self.in_features = original_linear.in_features
        self.active_pos = sorted(target_neurons)
        self.fixed_pos = [i for i in range(self.out_features) if i not in self.active_pos]

        # 参数验证
        if not all(0 <= idx < self.out_features for idx in self.active_pos):
            raise ValueError(f"激活索引必须在[0, {self.out_features-1}]范围内")
        if len(self.active_pos) != len(set(self.active_pos)):
            raise ValueError("激活索引包含重复值")

        # 分割权重矩阵
        W = original_linear.weight.data
        self.active_weight = nn.Parameter(W[self.active_pos].clone(), requires_grad=True)
        self.fixed_weight = nn.Parameter(W[self.fixed_pos].clone(), requires_grad=False)

        # 处理偏置项
        if original_linear.bias is not None:
            b = original_linear.bias.data
            self.active_bias = nn.Parameter(b[self.active_pos].clone(), requires_grad=True)
            self.fixed_bias = nn.Parameter(b[self.fixed_pos].clone(), requires_grad=False)
        else:
            self.register_parameter('active_bias', None)
            self.register_parameter('fixed_bias', None)

        # 预生成索引映射
        index_map = torch.empty(self.out_features, dtype=torch.long)
        index_map[self.active_pos] = torch.arange(len(self.active_pos))
        index_map[self.fixed_pos] = torch.arange(len(self.fixed_pos)) + len(self.active_pos)
        self.register_buffer('index_map', index_map)

    def forward(self, x):
        # 合并权重并进行矩阵运算
        weight = torch.cat([self.active_weight, self.fixed_weight], dim=0)
        output = torch.matmul(x, weight.transpose(-1, -2))
        
        # 添加合并后的偏置
        if self.active_bias is not None:
            bias = torch.cat([self.active_bias, self.fixed_bias], dim=0)
            output += bias.unsqueeze(0).unsqueeze(0)  # 广播偏置到所有batch和序列位置

        # 通过预生成索引重排输出
        return output.gather(
            dim=-1,
            index=self.index_map.view(1, 1, -1).expand(output.size(0), output.size(1), -1)
        )


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
import torch
import torch.nn as nn


class LoKILinear(nn.Module):
    def __init__(self, original_linear, target_neurons):
        super().__init__()
        self.out_features = original_linear.out_features
        self.in_features = original_linear.in_features
        self.active_pos = sorted(target_neurons)
        self.fixed_pos = [
            i for i in range(self.out_features) if i not in self.active_pos
        ]

        # 参数验证
        if not all(0 <= idx < self.out_features for idx in self.active_pos):
            raise ValueError(f"激活索引必须在[0, {self.out_features - 1}]范围内")
        if len(self.active_pos) != len(set(self.active_pos)):
            raise ValueError("激活索引包含重复值")
        self.active = nn.Linear(self.in_features, len(self.active_pos), bias=False)
        self.fixed = nn.Linear(self.in_features, len(self.fixed_pos), bias=False)
        # 分割权重矩阵
        W = original_linear.weight.data
        self.active.weight = nn.Parameter(W[self.active_pos].clone(), requires_grad=True)
        self.fixed.weight = nn.Parameter(W[self.fixed_pos].clone(), requires_grad=False)

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

        # 预生成索引映射
        index_map = torch.empty(self.out_features, dtype=torch.long)
        index_map[self.active_pos] = torch.arange(len(self.active_pos))
        index_map[self.fixed_pos] = torch.arange(len(self.fixed_pos)) + len(
            self.active_pos
        )
        self.register_buffer("index_map", index_map)

    def forward(self, x):
        active_out = self.active(x)  # 通过子模块计算激活部分
        fixed_out = self.fixed(x)    # 固定部分
        output = torch.cat([active_out, fixed_out], dim=-1)

        # 添加合并后的偏置
        if self.active_bias is not None:
            bias = torch.cat([self.active_bias, self.fixed_bias], dim=0)
            output += bias.unsqueeze(0).unsqueeze(0)  # 广播偏置到所有batch和序列位置

        # 通过预生成索引重排输出
        return output.gather(
            dim=-1,
            index=self.index_map.view(1, 1, -1).expand(
                output.size(0), output.size(1), -1
            ),
        )


class LoKILinear_i(nn.Module):
    def __init__(self, original_linear, target_neurons):
        super().__init__()
        self.out_features = original_linear.out_features
        self.in_features = original_linear.in_features
        self.active_pos = sorted(target_neurons)
        self.fixed_pos = [
            i for i in range(self.in_features) if i not in self.active_pos
        ]

        # 参数验证
        if not all(0 <= idx < self.in_features for idx in self.active_pos):
            raise ValueError(f"激活索引必须在[0, {self.in_features - 1}]范围内")
        if len(self.active_pos) != len(set(self.active_pos)):
            raise ValueError("激活索引包含重复值")

        # 分割权重矩阵的列
        W = original_linear.weight.data
        self.active_weight = nn.Parameter(
            W[:, self.active_pos].clone(), requires_grad=True
        )
        self.fixed_weight = nn.Parameter(
            W[:, self.fixed_pos].clone(), requires_grad=False
        )

        # 处理偏置项
        if original_linear.bias is not None:
            self.bias = nn.Parameter(
                original_linear.bias.data.clone(), requires_grad=True
            )
        else:
            self.bias = None

        # 生成输入特征重排索引
        self.register_buffer(
            "input_permutation",
            torch.tensor(self.active_pos + self.fixed_pos, dtype=torch.long),
        )

    def forward(self, x):
        # 合并权重矩阵
        weight = torch.cat([self.active_weight, self.fixed_weight], dim=1)

        # 重排输入特征顺序
        x_reordered = x.index_select(-1, self.input_permutation)

        # 矩阵乘法
        output = torch.matmul(x_reordered, weight.transpose(-1, -2))

        # 添加偏置项
        if self.bias is not None:
            output += self.bias.unsqueeze(0).expand_as(output)

        return output


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
        # isM=hasattr(mlp, 'down_proj')
        # isI = isinstance(mlp.down_proj, LoKILinear)
        # print(isM, isI, type(mlp.down_proj))
        if hasattr(mlp, "down_proj"):
            loki_layer = mlp.down_proj

            # 创建标准线性层
            original_linear = nn.Linear(
                in_features=loki_layer.in_features,
                out_features=loki_layer.out_features,
                bias=loki_layer.active_bias is not None,
                dtype=torch.bfloat16,
            )

            # 合并权重矩阵
            device = loki_layer.active_weight.device
            combined_weight = torch.zeros(
                (loki_layer.out_features, loki_layer.in_features),
                device=device,
                dtype=torch.bfloat16,
            )
            combined_weight[loki_layer.active_pos] = loki_layer.active_weight.data
            combined_weight[loki_layer.fixed_pos] = loki_layer.fixed_weight.data
            original_linear.weight.data = combined_weight

            # 合并偏置向量（如果存在）
            if loki_layer.active_bias is not None:
                combined_bias = torch.zeros(
                    loki_layer.out_features, device=device, dtype=torch.bfloat16
                )
                combined_bias[loki_layer.active_pos] = loki_layer.active_bias.data
                combined_bias[loki_layer.fixed_pos] = loki_layer.fixed_bias.data
                original_linear.bias.data = combined_bias

            # 保持设备一致性
            original_linear = original_linear.to(device)

            # 替换回原始结构
            setattr(mlp, "down_proj", original_linear)
            print(f"成功还原层{layer_idx}的down_proj")

    return model

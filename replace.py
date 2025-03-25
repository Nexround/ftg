from safetensors import safe_open
from transformers import AutoModelForCausalLM
from src.module.loki_linear import LoKILinear
import json
import torch

checkpoint_path = (
    "/cache/models/loki_reranker_qwen2_5-0-5b-5/checkpoint-456688/model.safetensors"
)
target_neurons_path = "target_neurons/Qwen2.5-0.5B-Instruct/5.json"
target_model = "/cache/models/loki_reranker_qwen2_5-0-5b-5/checkpoint-456688"

with open(target_neurons_path, "r", encoding="utf-8") as f:
    data = json.load(f)
trainable_neurons = list(data)
# 重新初始化原始模型结构
original_model = AutoModelForCausalLM.from_pretrained(target_model, torch_dtype=torch.bfloat16)


def merge_loki_weights(loki_layer, original_linear):
    # 合并权重矩阵
    merged_weight = torch.zeros_like(original_linear.weight.data)
    merged_weight[loki_layer.active_pos] = loki_layer.active_part.weight.data
    merged_weight[loki_layer.fixed_pos] = loki_layer.fixed_part.weight.data

    # 合并偏置项
    if original_linear.bias is not None:
        merged_bias = torch.zeros_like(original_linear.bias.data)
        merged_bias[loki_layer.active_pos] = loki_layer.active_bias.data
        merged_bias[loki_layer.fixed_pos] = loki_layer.fixed_bias.data

    # 加载参数到原始层
    original_linear.weight.data.copy_(merged_weight)
    print(torch.allclose(original_linear.weight.data[loki_layer.active_pos], loki_layer.active_part.weight.data, atol=1e-6))  # 应输出 True
    
    if original_linear.bias is not None:
        original_linear.bias.data.copy_(merged_bias)


# 加载检查点文件

# 遍历所有层还原参数
for layer_idx in range(original_model.config.num_hidden_layers):
    # 获取当前层的原始结构
    original_down_proj = original_model.model.layers[layer_idx].mlp.down_proj

    # 加载LoKI层参数
    with safe_open(checkpoint_path, framework="pt") as f:
        # 创建临时LoKI层用于加载参数
        loki_layer = LoKILinear(
            original_down_proj, target_neurons=trainable_neurons[layer_idx]
        )
        loki_layer.load_state_dict(
            {
                "active_part.weight": f.get_tensor(
                    f"model.layers.{layer_idx}.mlp.down_proj.active_part.weight"
                ),
                "fixed_part.weight": f.get_tensor(
                    f"model.layers.{layer_idx}.mlp.down_proj.fixed_part.weight"
                ),
                # "active_bias": f.get_tensor(
                #     f"model.layers.{layer_idx}.mlp.down_proj.active_bias"
                # ),
                # "fixed_bias": f.get_tensor(
                #     f"model.layers.{layer_idx}.mlp.down_proj.fixed_bias"
                # ),
            },
            strict=True,
        )

    # 合并参数到原始层
    merge_loki_weights(loki_layer, original_down_proj)

# 保存还原后的模型
original_model.save_pretrained("/cache/models/loki_reranker_qwen2_5-0-5b-5_real")
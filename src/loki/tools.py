from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import torch
from .loki_qwen_config import LoKIQwen2Config
from .loki_llama_config import LoKILlamaConfig

from .loki_qwen_model import LoKIQwen2ForCausalLM, LoKIQwen2ForCausalLM_i
from .loki_llama_model import LoKILlamaForCausalLM
from .loki_linear import LoKILinear, LoKILinear_i

from safetensors import safe_open
from pathlib import Path


def create_and_save_loki_model(
    target_neurons_path: str,
    save_dir: str,
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
    torch_dtype: torch.dtype = torch.bfloat16,
) -> None:
    """
    创建并保存LoKI自定义模型

    参数:
    target_neurons_path: 目标神经元配置文件路径
    save_dir: 模型保存目录
    model_name: 基础模型名称，默认为"Qwen/Qwen2.5-0.5B-Instruct"
    torch_dtype: 模型精度，默认为torch.bfloat16
    """
    # 加载目标神经元配置
    with open(target_neurons_path, "r", encoding="utf-8") as f:
        target_neurons = json.load(f)

    # 加载原始模型
    original_model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch_dtype
    )

    # 注册自定义模型类到AutoClass
    LoKIQwen2ForCausalLM.register_for_auto_class("AutoModelForCausalLM")
    LoKIQwen2Config.register_for_auto_class()

    # 创建LoKI配置
    loki_config = LoKIQwen2Config.from_pretrained(model_name)
    loki_config.target_neurons = target_neurons

    # 加载LoKI模型
    loki_model = LoKIQwen2ForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_name,
        config=loki_config,
        torch_dtype=torch_dtype,
    )

    # 保存模型和tokenizer
    loki_model.save_pretrained(save_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(save_dir)

    # 保存原始模型参数（可选）
    original_model.save_pretrained(save_dir, is_main_process=False)

    print("Done.")
    
def create_and_save_loki_model_llama(
    target_neurons_path: str,
    save_dir: str,
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
    torch_dtype: torch.dtype = torch.bfloat16,
) -> None:
    """
    创建并保存LoKI自定义模型

    参数:
    target_neurons_path: 目标神经元配置文件路径
    save_dir: 模型保存目录
    model_name: 基础模型名称，默认为"Qwen/Qwen2.5-0.5B-Instruct"
    torch_dtype: 模型精度，默认为torch.bfloat16
    """
    # 加载目标神经元配置
    with open(target_neurons_path, "r", encoding="utf-8") as f:
        target_neurons = json.load(f)

    # 加载原始模型
    original_model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch_dtype
    )

    # 注册自定义模型类到AutoClass
    LoKILlamaForCausalLM.register_for_auto_class("AutoModelForCausalLM")
    LoKILlamaConfig.register_for_auto_class()

    # 创建LoKI配置
    loki_config = LoKILlamaConfig.from_pretrained(model_name)
    loki_config.target_neurons = target_neurons

    # 加载LoKI模型
    loki_model = LoKILlamaForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_name,
        config=loki_config,
        torch_dtype=torch_dtype,
    )

    # 保存模型和tokenizer
    loki_model.save_pretrained(save_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(save_dir)

    # 保存原始模型参数（可选）
    original_model.save_pretrained(save_dir, is_main_process=False)

    print("Done.")


def create_and_save_loki_model_i(
    target_neurons_path: str,
    save_dir: str,
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
    torch_dtype: torch.dtype = torch.bfloat16,
) -> None:
    """
    创建并保存LoKI自定义模型

    参数:
    target_neurons_path: 目标神经元配置文件路径
    save_dir: 模型保存目录
    model_name: 基础模型名称，默认为"Qwen/Qwen2.5-0.5B-Instruct"
    torch_dtype: 模型精度，默认为torch.bfloat16
    """
    # 加载目标神经元配置
    with open(target_neurons_path, "r", encoding="utf-8") as f:
        target_neurons = json.load(f)

    # 加载原始模型
    original_model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch_dtype
    )

    # 注册自定义模型类到AutoClass
    LoKIQwen2ForCausalLM_i.register_for_auto_class("AutoModelForCausalLM")
    LoKIQwen2Config.register_for_auto_class()

    # 创建LoKI配置
    loki_config = LoKIQwen2Config.from_pretrained(model_name)
    loki_config.target_neurons = target_neurons

    # 加载LoKI模型
    loki_model = LoKIQwen2ForCausalLM_i.from_pretrained(
        pretrained_model_name_or_path=model_name,
        config=loki_config,
        torch_dtype=torch_dtype,
    )

    # 保存模型和tokenizer
    loki_model.save_pretrained(save_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(save_dir)

    # 保存原始模型参数（可选）
    original_model.save_pretrained(save_dir, is_main_process=False)

    print("Done.")


def merge_loki_weights(loki_layer, original_linear):
    # 合并权重矩阵
    merged_weight = torch.zeros_like(original_linear.weight.data)
    merged_weight[loki_layer.active_pos] = loki_layer.active_weight.data
    merged_weight[loki_layer.fixed_pos] = loki_layer.fixed_weight.data

    # 合并偏置项
    if original_linear.bias is not None:
        merged_bias = torch.zeros_like(original_linear.bias.data)
        merged_bias[loki_layer.active_pos] = loki_layer.active_bias.data
        merged_bias[loki_layer.fixed_pos] = loki_layer.fixed_bias.data

    # 加载参数到原始层
    original_linear.weight.data.copy_(merged_weight)
    if original_linear.bias is not None:
        original_linear.bias.data.copy_(merged_bias)


def restore_loki_model(
    target_neurons_path: str,
    model_path: str,
    original_model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
    output_path: str = "/cache/models/loki_reranker_qwen2_5-0-5b-10_real",
    torch_dtype: torch.dtype = torch.bfloat16,
):
    with open(target_neurons_path, "r", encoding="utf-8") as f:
        target_neurons = json.load(f)
    original_model = AutoModelForCausalLM.from_pretrained(
        original_model_name, torch_dtype=torch_dtype
    )
    safe_tensor_path = Path(model_path, "model.safetensors")
    # 遍历所有层还原参数
    for layer_idx in range(original_model.config.num_hidden_layers):
        # 获取当前层的原始结构
        if not target_neurons[layer_idx]:
            continue
        original_down_proj = original_model.model.layers[layer_idx].mlp.down_proj

        # 加载LoKI层参数
        with safe_open(safe_tensor_path, framework="pt") as f:
            # 创建临时LoKI层用于加载参数
            loki_layer = LoKILinear(
                original_down_proj, target_neurons=target_neurons[layer_idx]
            )
            loki_layer.load_state_dict(
                {
                    "active_weight": f.get_tensor(
                        f"model.layers.{layer_idx}.mlp.down_proj.active_weight"
                    ),
                    "fixed_weight": f.get_tensor(
                        f"model.layers.{layer_idx}.mlp.down_proj.fixed_weight"
                    ),
                    "index_map": loki_layer.index_map,
                },
                strict=True,
            )

        # 合并参数到原始层
        merge_loki_weights(loki_layer, original_down_proj)

    # 保存还原后的模型
    original_model.save_pretrained(output_path)
    tokenizer = AutoTokenizer.from_pretrained(original_model_name)
    tokenizer.save_pretrained(output_path)
    

from glob import glob

def restore_loki_model_(
    target_neurons_path: str,
    model_path: str,
    original_model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
    output_path: str = "/cache/models/loki_reranker_qwen2_5-0-5b-10_real",
    torch_dtype: torch.dtype = torch.bfloat16,
):
    # 加载目标神经元配置
    with open(target_neurons_path, "r", encoding="utf-8") as f:
        target_neurons = json.load(f)
    
    # 加载原始模型
    original_model = AutoModelForCausalLM.from_pretrained(
        original_model_name, torch_dtype=torch_dtype
    )
    
    # 加载所有分片文件 ----------------------------------------------------
    # 获取分片文件列表并按顺序排序
    shard_files = sorted(glob(f"{model_path}/model-*-of-*.safetensors"))
    
    # 合并所有分片的张量到内存
    tensors = {}
    for shard_file in shard_files:
        with safe_open(shard_file, framework="pt") as f:
            for key in f.keys():
                tensors[key] = f.get_tensor(key)
    # ---------------------------------------------------------------------

    # 遍历所有层还原参数
    for layer_idx in range(original_model.config.num_hidden_layers):
        if not target_neurons[layer_idx]:
            continue
        
        original_down_proj = original_model.model.layers[layer_idx].mlp.down_proj

        # 从合并的张量中加载参数 ----------------------------------------------
        loki_layer = LoKILinear(
            original_down_proj, target_neurons=target_neurons[layer_idx]
        )
        
        # 构造参数键名
        active_weight_key = f"model.layers.{layer_idx}.mlp.down_proj.active_weight"
        fixed_weight_key = f"model.layers.{layer_idx}.mlp.down_proj.fixed_weight"
        
        # 从预加载的张量字典中获取参数
        loki_layer.load_state_dict(
            {
                "active_weight": tensors[active_weight_key],
                "fixed_weight": tensors[fixed_weight_key],
                "index_map": loki_layer.index_map,  # 该参数由LoKILinear内部生成
            },
            strict=True,
        )
        # ---------------------------------------------------------------------

        # 合并参数到原始层
        merge_loki_weights(loki_layer, original_down_proj)

    # 保存还原后的模型
    original_model.save_pretrained(output_path)
    tokenizer = AutoTokenizer.from_pretrained(original_model_name)
    tokenizer.save_pretrained(output_path)


def restore_loki_model_i(
    target_neurons_path: str,
    model_path: str,
    original_model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
    output_path: str = "/cache/models/loki_reranker_qwen2_5-0-5b-10_real",
    torch_dtype: torch.dtype = torch.bfloat16,
):
    with open(target_neurons_path, "r", encoding="utf-8") as f:
        target_neurons = json.load(f)

    # 加载原始模型
    original_model = AutoModelForCausalLM.from_pretrained(
        original_model_name, torch_dtype=torch_dtype
    )

    # 参数合并函数
    def merge_weights(loki_layer, original_layer):
        """合并列分割的权重到原始层"""
        # 合并active和fixed列
        merged_weights = torch.cat(
            [loki_layer.active_weight, loki_layer.fixed_weight], dim=1
        )

        # 生成逆排列索引
        permuted_indices = loki_layer.active_pos + loki_layer.fixed_pos
        inv_perm = torch.argsort(torch.tensor(permuted_indices))

        # 还原权重矩阵原始顺序
        original_layer.weight.data.copy_(merged_weights[:, inv_perm])

        # 还原偏置项
        if original_layer.bias is not None:
            original_layer.bias.data.copy_(loki_layer.bias.data)

    # 加载安全张量文件
    safe_tensor_path = Path(model_path) / "model.safetensors"

    with safe_open(safe_tensor_path, framework="pt") as f:
        for layer_idx in range(original_model.config.num_hidden_layers):
            if not target_neurons[layer_idx]:
                continue

            # 获取原始层引用
            original_proj = original_model.model.layers[layer_idx].mlp.down_proj

            try:
                # 初始化LoKI层并加载参数
                loki_layer = LoKILinear_i(
                    original_proj, target_neurons=target_neurons[layer_idx]
                )

                # 构建状态字典
                state_dict = {
                    "active_weight": f.get_tensor(
                        f"model.layers.{layer_idx}.mlp.down_proj.active_weight"
                    ),
                    "fixed_weight": f.get_tensor(
                        f"model.layers.{layer_idx}.mlp.down_proj.fixed_weight"
                    ),
                }

                loki_layer.load_state_dict(state_dict, strict=False)

                # 合并参数到原始层
                merge_weights(loki_layer, original_proj)

            except KeyError as e:
                print(f"跳过第{layer_idx}层: {str(e)}")
                continue

    # 保存还原后的模型
    original_model.save_pretrained(output_path)
    tokenizer = AutoTokenizer.from_pretrained(original_model_name)
    tokenizer.save_pretrained(output_path)
    print(f"模型已保存至: {output_path}")


def set_zero_weights(
    target_neurons_path: str,
    output_path: str,
    original_model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
    torch_dtype: torch.dtype = torch.bfloat16,
):
    with open(target_neurons_path, "r", encoding="utf-8") as f:
        target_neurons = json.load(f)
    original_model = AutoModelForCausalLM.from_pretrained(
        original_model_name, torch_dtype=torch_dtype
    )
    for layer_idx in range(original_model.config.num_hidden_layers):
        # 获取当前层的原始结构
        original_down_proj = original_model.model.layers[layer_idx].mlp.down_proj
        weight = original_down_proj.weight
        # 遍历要置零的神经元索引
        with torch.no_grad():
            for neuron_idx in target_neurons[layer_idx]:
                if neuron_idx < weight.shape[0]:  # 确保索引有效
                    weight[neuron_idx, :] = 0.0  # 将该神经元的权重置零

    # 保存还原后的模型
    original_model.save_pretrained(output_path)
    tokenizer = AutoTokenizer.from_pretrained(original_model_name)
    tokenizer.save_pretrained(output_path)

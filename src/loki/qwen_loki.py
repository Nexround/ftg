from transformers import Qwen2Config, Qwen2ForCausalLM
from .loki_linear import LoKILinear


class LoKIQwen2Config(Qwen2Config):
    def __init__(self, target_neurons, **kwargs):
        super().__init__(**kwargs)
        self.target_neurons = target_neurons


class LoKIQwen2ForCausalLM(Qwen2ForCausalLM):
    def __init__(self, config, target_neurons=None):  # 修改1：明确接收config参数
        super().__init__(config)  # 修改2：正确初始化父类
        # self.target_neurons = target_neurons or [None] * config.num_hidden_layers  # 修改3：处理默认值
        # TODO: 修正
        target_neurons_path = "target_neurons/Qwen2.5-0.5B-Instruct/10.json"

        import json
        with open(target_neurons_path, "r", encoding="utf-8") as f:
            self.target_neurons = json.load(f)
        # 确保target_neurons长度匹配
        if len(self.target_neurons) != config.num_hidden_layers:
            raise ValueError(
                f"target_neurons长度({len(target_neurons)})必须等于num_hidden_layers({config.num_hidden_layers})"
            )
        # self.post_init()  # 确保正确初始化
        
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, config = None, cache_dir = None, ignore_mismatched_sizes = False, force_download = False, local_files_only = False, token = None, revision = "main", use_safetensors = None, weights_only = True, **kwargs):
        model = super().from_pretrained(pretrained_model_name_or_path, *model_args, config=config, cache_dir=cache_dir, ignore_mismatched_sizes=ignore_mismatched_sizes, force_download=force_download, local_files_only=local_files_only, token=token, revision=revision, use_safetensors=use_safetensors, weights_only=weights_only, **kwargs)
        for param in model.model.parameters():
            param.requires_grad = False
        for param in model.lm_head.parameters():
            param.requires_grad = False
        model.replace_all_target_linear_qwen()  # 修改4：移除冗余参数传递
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(f"Parameter: {name}, Shape: {param.shape}")
        trainable_param_count = sum(
            param.numel() for param in model.parameters() if param.requires_grad
        )
        print(f"Total Trainable Parameters: {trainable_param_count}")
        return model
    

    def replace_all_target_linear_qwen(self):
        for layer_idx in range(self.config.num_hidden_layers):
            # 获取目标层设备信息
            original_layer = self.model.layers[layer_idx].mlp.down_proj
            device = original_layer.weight.device
            
            # 初始化LoKILinear
            new_layer = LoKILinear(
                original_linear=original_layer,
                target_neurons=self.target_neurons[layer_idx]  # 自定义参数
            ).to(device)
            
            # 安全替换（保持梯度追踪）
            setattr(self.model.layers[layer_idx].mlp, 'down_proj', new_layer)
            print(f"成功替换层{layer_idx}")
            # self.model.layers[layer_idx].mlp.down_proj = new_layer  # 修改5：直接使用属性赋值

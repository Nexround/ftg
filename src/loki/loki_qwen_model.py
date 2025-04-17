from transformers import Qwen2ForCausalLM
from .loki_qwen_config import LoKIQwen2Config
from .loki_linear import LoKILinear
import torch


class LoKIQwen2ForCausalLM(Qwen2ForCausalLM):
    config_class = LoKIQwen2Config
    def __init__(self, config):  # 修改1：明确接收config参数
        super().__init__(config)  # 修改2：正确初始化父类

        self.target_neurons = config.target_neurons  # 从配置中读取参数

        # 确保target_neurons长度匹配
        if len(self.target_neurons) != config.num_hidden_layers:
            raise ValueError(
                f"target_neurons长度({len(self.target_neurons)})必须等于num_hidden_layers({config.num_hidden_layers})"
            )

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path,
        *model_args,
        config=None,
        cache_dir=None,
        ignore_mismatched_sizes=False,
        force_download=False,
        local_files_only=False,
        token=None,
        revision="main",
        use_safetensors=None,
        weights_only=True,
        **kwargs,
    ):
        model = super().from_pretrained(
            pretrained_model_name_or_path,
            *model_args,
            config=config,
            cache_dir=cache_dir,
            ignore_mismatched_sizes=ignore_mismatched_sizes,
            force_download=force_download,
            local_files_only=local_files_only,
            token=token,
            revision=revision,
            use_safetensors=use_safetensors,
            weights_only=weights_only,
            **kwargs,
        )
        for param in model.model.parameters():
            param.requires_grad = False
        for param in model.lm_head.parameters():
            param.requires_grad = False
        # model.apply_selective_ffn_gradient_masking()  # 修改4：移除冗余参数传递
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
            if len(self.target_neurons[layer_idx]) > 0:
                # 初始化LoKILinear
                new_layer = LoKILinear(
                    original_linear=original_layer,
                    target_neurons=self.target_neurons[layer_idx],  # 自定义参数
                ).to(device)

                # 安全替换（保持梯度追踪）
                setattr(self.model.layers[layer_idx].mlp, "down_proj", new_layer)
                print(f"成功替换层{layer_idx}")
            # self.model.layers[layer_idx].mlp.down_proj = new_layer  # 修改5：直接使用属性赋值

    def apply_selective_ffn_gradient_masking(self):
        hooks = []

        for layer_idx, neuron_indices in enumerate(self.target_neurons):
            target_mlp = self.model.layers[layer_idx].mlp.down_proj
            target_mlp.weight.requires_grad = True

            def _hook(grad):
                mask = torch.zeros_like(grad)
                mask[neuron_indices, :] = 1  # 只允许指定神经元的梯度
                return grad * mask

            hooks.append(target_mlp.weight.register_hook(_hook))

        return hooks  # 返回 hooks 以便后续管理和清理

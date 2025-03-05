from transformers import Qwen2ForCausalLM
import torch
import torch.nn.functional as F
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor

class CustomQwen2ForCausalLM(Qwen2ForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self._intermediate_activations = []
        self._original_activations = []
        self._partitioning_activations = []
        self._partitioning_step = []
        self._partitioning_logits = []
        self._integrated_gradients = []
        self._args = None
        self._kwargs = None
        self._new_dict = None
        # self._memory_net = getattr(self, "mlp.down_proj")
        # 冻结所有参数，仅保留需要的梯度
        for param in self.model.parameters():
            param.requires_grad = False
        for layer in self.model.layers:
            layer.mlp.down_proj.weight.requires_grad = True

    def forward(self, target_token_idx, *args, **kwargs):
        self._args = args
        self._kwargs = kwargs
        keys_to_remove = ["input_ids", "attention_mask"]
        self._new_dict = {k: v for k, v in kwargs.items() if k not in keys_to_remove}

        # Hook捕获中间激活
        def hook_fn(module, input, output):
            self._intermediate_activations.append(output[:, target_token_idx, :])

        hooks = []
        for layer in self.model.layers:
            hooks.append(layer.mlp.down_proj.register_forward_hook(hook_fn))
        
        outputs = super().forward(*args, **kwargs).logits[:, target_token_idx, :]
        # 只返回指定位置的logits
        for hook in hooks:
            hook.remove()
        
        return outputs

    def forward_with_partitioning(self, target_token_idx, times):
        # 生成所有层的分区激活
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.generate_partitioning, vec, times) for vec in self._intermediate_activations]
            for future in concurrent.futures.as_completed(futures):
                partitioning, step = future.result()
                self._partitioning_activations.append(partitioning)
                self._partitioning_step.append(step)

        # 准备批量输入（总样本数 = 层数 * 分区次数）
        num_layers = self.config.num_hidden_layers
        total_samples = num_layers * times
        input_ids = self._kwargs["input_ids"].repeat(total_samples, 1)
        attention_mask = self._kwargs["attention_mask"].repeat(total_samples, 1)

        # 注册所有层的Hook
        hooks = []
        for layer_idx in range(num_layers):
            layer_activation = self._partitioning_activations[layer_idx]
            hook = self._create_batch_hook(
                layer_idx, target_token_idx, layer_activation, times,target= self.model.layers[layer_idx].mlp.down_proj
            )
            hooks.append(hook)

        # 单次前向传播
        outputs = self.model(input_ids, attention_mask, **self._new_dict)
        all_logits = self.lm_head(outputs.last_hidden_state[:, target_token_idx, :])

        # 按层分割结果
        self._partitioning_logits = [
            all_logits[i*times : (i+1)*times] 
            for i in range(num_layers)
        ]

        # 清理Hook
        for hook in hooks:
            hook.remove()
        
        return self._partitioning_logits

    def _create_batch_hook(self, layer_idx, token_idx, activations, times, target):
        
        def hook_fn(module, input, output):
            start = layer_idx * times
            end = start + times
            output = output.clone()
            output[start:end, token_idx] = activations
            return output
        
        return target.register_forward_hook(hook_fn)

    def generate_partitioning(self, vector, times):
        baseline = torch.zeros_like(vector)
        step = (vector - baseline) / times
        partitioning = torch.cat([baseline + step*i for i in range(times)], dim=0)
        return partitioning, step[0].detach().cpu()

    def _compute_ig_for_layer(self, i, target_label):
        prob = F.softmax(self._partitioning_logits[i], dim=1)
        target_label_logits = prob[:, target_label]
        (gradient,) = torch.autograd.grad(
            target_label_logits,
            self._partitioning_activations[i],
            grad_outputs=torch.ones_like(target_label_logits),
            retain_graph=True
        )
        gradient = gradient.detach().cpu()
        return gradient.sum(dim=0) * self._partitioning_step[i]

    def calculate_integrated_gradients(self, target_label):
        self._integrated_gradients = [None] * self.config.num_hidden_layers
        with ThreadPoolExecutor() as executor:
            futures = {executor.submit(self._compute_ig_for_layer, i, target_label): i 
                    for i in range(self.config.num_hidden_layers)}
            for future in concurrent.futures.as_completed(futures):
                i = futures[future]
                self._integrated_gradients[i] = future.result()
        return self._integrated_gradients

    def clean(self):
        attrs = [
            '_intermediate_activations', '_partitioning_activations',
            '_partitioning_step', '_partitioning_logits', '_integrated_gradients'
        ]
        for attr in attrs:
            getattr(self, attr).clear()
        self._args = None
        self._kwargs = None


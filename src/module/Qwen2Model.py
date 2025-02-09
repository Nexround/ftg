from transformers import Qwen2ForCausalLM
import torch
import torch.nn.functional as F

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
        
        # 冻结所有参数，仅保留需要的梯度
        for param in self.model.parameters():
            param.requires_grad = False
        for layer in self.model.layers:
            layer.mlp.gate_proj.weight.requires_grad = True

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
            hooks.append(layer.mlp.gate_proj.register_forward_hook(hook_fn))
        
        outputs = super().forward(*args, **kwargs).logits[:, target_token_idx, :]
        
        for hook in hooks:
            hook.remove()
        
        return outputs

    def forward_with_partitioning(self, target_token_idx, times):
        # 生成所有层的分区激活
        self._partitioning_activations = []
        self._partitioning_step = []
        for vec in self._intermediate_activations:
            partitioning, step = self.generate_partitioning(vec, times)
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
                layer_idx, target_token_idx, layer_activation, times,target= self.model.layers[layer_idx].mlp.gate_proj
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

    def calculate_integrated_gradients(self, target_label):
        self._integrated_gradients = []
        for i in range(self.config.num_hidden_layers):
            prob = F.softmax(self._partitioning_logits[i], dim=1)
            o = prob[:, target_label]
            (gradient,) = torch.autograd.grad(
                o, 
                self._partitioning_activations[i], 
                grad_outputs=torch.ones_like(o),
                retain_graph=True
            )
            ig = (gradient.sum(dim=0).cpu() * self._partitioning_step[i]).cpu()
            self._integrated_gradients.append(ig)
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


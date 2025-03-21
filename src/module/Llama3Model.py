from transformers import LlamaForCausalLM
import torch
import torch.nn.functional as F


class CustomLlamaForCausalLM(LlamaForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self._intermediate_activations = []
        self._partitioning_activations = []
        self._partitioning_step = []
        self._partitioning_logits = []
        self.integrated_gradients = [None] * self.config.num_hidden_layers
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
        for layer in self.model.layers:
            layer.mlp.down_proj.weight.requires_grad = True
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

    def forward_with_partitioning(self, target_token_idx, times, predicted_label):
        # 生成所有层的分区激活
        for param in self.model.parameters():
            param.requires_grad = False

        for vec in self._intermediate_activations:
            partitioning, step = self.generate_partitioning(vec, times)
            self._partitioning_activations.append(partitioning)
            self._partitioning_step.append(step)

        num_layers = self.config.num_hidden_layers

        # 逐层进行批量推理
        for layer_idx in range(num_layers):
            self.model.layers[layer_idx].mlp.down_proj.weight.requires_grad = True

            # 准备当前层的输入（重复times次）
            layer_input_ids = self._kwargs["input_ids"].repeat(times, 1)
            layer_attention_mask = self._kwargs["attention_mask"].repeat(times, 1)

            # 获取当前层的激活
            layer_activation = self._partitioning_activations[layer_idx]

            # 注册当前层的Hook
            hook = self._create_layer_hook(
                target_token_idx=target_token_idx,
                activations=layer_activation,
                target=self.model.layers[layer_idx].mlp.down_proj,
            )

            try:
                # 尝试批量推理
                outputs = self.model(
                    layer_input_ids, layer_attention_mask, **self._new_dict
                )
                layer_logits = self.lm_head(
                    outputs.last_hidden_state[:, target_token_idx, :]
                )
            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    # 内存不足时降级为逐样本处理
                    print(
                        f"Layer {layer_idx} OOM, falling back to per-sample processing"
                    )
                    layer_logits = []
                    if "hook" in locals():
                        hook.remove()
                    # 清空缓存
                    torch.cuda.empty_cache()

                    # 逐个样本处理
                    for i in range(times):
                        # 准备单个样本
                        single_input_ids = self._kwargs["input_ids"][i].unsqueeze(0)
                        single_attention_mask = self._kwargs["attention_mask"][
                            i
                        ].unsqueeze(0)
                        # 注册当前样本的Hook
                        hook = self._create_layer_hook(
                            target_token_idx=target_token_idx,
                            activations=layer_activation[i],  # 取对应的激活部分
                            target=self.model.layers[layer_idx].mlp.down_proj,
                        )
                        # 处理单个样本
                        outputs = self.model(
                            single_input_ids, single_attention_mask, **self._new_dict
                        )
                        single_logits = self.lm_head(
                            outputs.last_hidden_state[:, target_token_idx, :]
                        )
                        layer_logits.append(single_logits)

                        # 清理
                        hook.remove()
                        torch.cuda.empty_cache()

                    # 堆叠结果
                    layer_logits = torch.cat(layer_logits)
                else:
                    raise e

            # 保存结果并清理
            self._partitioning_logits.append(layer_logits)
            if "hook" in locals():  # 确保清理在try块中创建的hook
                hook.remove()
            self._compute_ig_for_layer(layer_idx, predicted_label)
            self.model.layers[layer_idx].mlp.down_proj.weight.requires_grad = False
            for param in self.parameters():
                if param.grad is not None:
                    print(param.grad)

        return self._partitioning_logits

    def _create_layer_hook(self, target_token_idx, activations, target):
        def hook_fn(module, input, output):
            output = output.clone()
            # 直接替换当前batch所有样本的对应位置
            output[:, target_token_idx] = activations
            return output

        return target.register_forward_hook(hook_fn)

    def generate_partitioning(self, vector, times):
        baseline = torch.zeros_like(vector)
        step = (vector - baseline) / times
        partitioning = torch.cat([baseline + step * i for i in range(times)], dim=0)
        return partitioning, step[0].detach().cpu()

    def _compute_ig_for_layer(self, i, target_label):
        prob = F.softmax(self._partitioning_logits[i], dim=1)
        # self._partitioning_logits[i] = None
        # prob = torch.argmax(self._partitioning_logits[i], dim=1)
        target_label_logits = prob[:, target_label]
        (gradient,) = torch.autograd.grad(
            target_label_logits,
            self._partitioning_activations[i],
            grad_outputs=torch.ones_like(target_label_logits),
            # create_graph=True,
            # retain_graph=True,
        )
        # dot = make_dot(gradient)
        # dot.attr(dpi="300")
        # dot.render("test", format="pdf")

        gradient = gradient.detach().cpu()
        with torch.no_grad():
            self.integrated_gradients[i] = (
                gradient.sum(dim=0) * self._partitioning_step[i]
            )

    def clean(self):
        attrs = [
            "_intermediate_activations",
            "_partitioning_activations",
            "_partitioning_step",
            "_partitioning_logits",
            "integrated_gradients",
        ]
        for attr in attrs:
            getattr(self, attr).clear()
        self.integrated_gradients = [None] * self.config.num_hidden_layers
        self._args = None
        self._kwargs = None


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


            # 获取当前层的激活
            layer_activation = self._partitioning_activations[layer_idx]

            layer_logits = []

            # 逐个样本处理
            for i in range(times):
                # 准备单个样本
                single_input_ids = self._kwargs["input_ids"]
                single_attention_mask = self._kwargs["attention_mask"]
                single_layer_activation = layer_activation[i].unsqueeze(0)
                # 注册当前样本的Hook
                hook = self._create_layer_hook(
                    target_token_idx=target_token_idx,
                    activations=single_layer_activation,  # 取对应的激活部分
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
                prob = F.softmax(single_logits, dim=1)

                target_label_logits = prob[:, predicted_label]
                print(target_label_logits.grad_fn)
                print(single_layer_activation.grad_fn)
                print(target_label_logits.shape)
                print(single_layer_activation.shape)
                
                (gradient,) = torch.autograd.grad(
                    target_label_logits,
                    single_layer_activation,
                    grad_outputs=torch.ones_like(target_label_logits),
                    # create_graph=True,
                    # retain_graph=True,
                )
                # dot = make_dot(gradient)
                # dot.attr(dpi="300")
                # dot.render("test", format="pdf")

                gradient = gradient.detach().cpu()

                # 清理
                hook.remove()
                print("gradient",gradient.shape)
                print("self._partitioning_step",self._partitioning_step[layer_idx].shape)
                with torch.no_grad():
                    self.integrated_gradients[layer_idx] = torch.zeros_like(gradient.squeeze(0))
                    self.integrated_gradients[layer_idx] += gradient.squeeze(0) * self._partitioning_step[layer_idx]

            self.model.layers[layer_idx].mlp.down_proj.weight.requires_grad = False


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

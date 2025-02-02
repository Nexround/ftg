from transformers import Qwen2ForCausalLM
import torch
import torch.nn.functional as F


class CustomQwen2ForCausalLM(Qwen2ForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self._intermediate_activations = []  # 逐层的每个token在ffn层的输出
        self._original_activations = []
        self._partitioning_activations = []
        self._partitioning_step = []
        self._partitioning_last_hidden_state_last_pos = []
        self._PARTITIONING_TIMES = 6
        self._partitioning_logits = []
        # self._partitioning_logits是last_hidden_state经过cls层的输出
        self._integrated_gradients = []
        self._temporary_hooks = []
        self._args = None
        self._kwargs = None
        self._new_dict = None
        for param in self.model.parameters():
            param.requires_grad = False
        for layer in range(self.config.num_hidden_layers):
            self.model.layers[layer].mlp.gate_proj.weight.requires_grad = True
            # self.model.layers[layer].mlp.down_proj.weight.requires_grad = True

    def forward(self, target_token_idx, *args, **kwargs):
        """
        保存现有参数以用于forward_with_partitioning
        """
        self._args = args
        self._kwargs = kwargs
        keys_to_remove = ["input_ids", "attention_mask"]

        self._new_dict = {
            key: value
            for key, value in self._kwargs.items()
            if key not in keys_to_remove
        }

        # Hook to capture intermediate activations
        def hook_fn(module, input, output):
            # 如果 _intermediate_activations 不为空，则清空它
            # if len(self._intermediate_activations) == self.config.num_hidden_layers:
            #     self._intermediate_activations.clear()

            # 添加新的激活到 _intermediate_activations
            self._intermediate_activations.append(
                output[:, target_token_idx, :]
            )  # torch.Size([1, token_len, 3072])

        # Register hooks on intermediate layers
        hooks = []
        for layer in self.model.layers:
            hooks.append(layer.mlp.down_proj.register_forward_hook(hook_fn))

        # Forward pass
        outputs = super().forward(*args, **kwargs).logits[:, target_token_idx, :]
        # Remove hooks after forward pass
        for hook in hooks:
            hook.remove()

        return outputs

    def forward_with_partitioning(self, target_token_idx):
        """
        target_pos token idx
        """
        # 批量化处理提高性能
        batch_size = self._PARTITIONING_TIMES
        # kwargs = [self._kwargs for _ in range(batch_size)]

        input_ids = self._kwargs["input_ids"].repeat(batch_size, 1)
        attention_mask = self._kwargs["attention_mask"].repeat(batch_size, 1)
        """
        找到对应token在每层的激活值，然后
        对每层的激活值进行间隔操作，并逐层记录
        """
        for target_vector in self._intermediate_activations:
            partitioning, step = self.generate_partitioning(
                target_vector, self._PARTITIONING_TIMES
            )
            self._partitioning_activations.append(partitioning)
            self._partitioning_step.append(step.detach())
            # self._partitioning_activations.append(partitioning.detach().cpu())
            # self._partitioning_step.append(step.detach().cpu())

        """
        
        """
        for layer_idx in range(self.config.num_hidden_layers):
            # for idx in range(self._PARTITIONING_TIMES):
            hook = self.modify_ffn_activation(
                layer_idx,
                target_token_idx,
                self._partitioning_activations[layer_idx],
            )
            self._temporary_hooks.append(hook)
            outputs = self.model.forward(input_ids, attention_mask, **self._new_dict)
            # outputs = super().forward(input_ids, attention_mask)
            # outputs = super().forward(*self._args, **self._kwargs)
            self._partitioning_last_hidden_state_last_pos.append(
                outputs.last_hidden_state[:, target_token_idx, :]
            )  # logits不可使用detach，否则会导致梯度丢失
            # self._partitioning_logits.append(
            #     outputs.logits
            # )  # logits不可使用detach，否则会导致梯度丢失
            del outputs
            # torch.cuda.empty_cache()
            self._partitioning_logits.append(
                self.lm_head(self._partitioning_last_hidden_state_last_pos[layer_idx])
            )  # 逐层地使用ffn激活值进行计算
        return self._partitioning_logits

    def modify_ffn_activation(self, layer_idx, target_token_idx, new_activation):
        """
        Modifies the hidden activations of a specific FFN layer at a specific position in the model.
        Stores the original activations before modifying.

        Args:
            layer_idx (int): Index of the transformer layer to modify (0-indexed).
            new_activation (torch.Tensor): The new activation values with shape matching the FFN layer output
                                            (e.g., [intermediate_size]).

        目标是直接一次性修改batch_size个样本的激活值
        """
        if not isinstance(new_activation, torch.Tensor):
            raise ValueError("new_activation must be a torch.Tensor.")

        def hook_fn(module, input, output):
            # Store original activations before modification
            # self._original_activations.append(output)

            # Ensure the shape matches
            # batch_idx, seq_idx = (
            #     target_position
            #     if isinstance(target_position, tuple)
            #     else (0, target_position)
            # )
            # 简易补丁
            if new_activation.shape != output[:, target_token_idx, :].shape:
                raise ValueError(
                    f"Shape mismatch: Expected {output[:,target_token_idx,:].shape}, "
                    f"but got {new_activation.shape}."
                )

            # Modify the activation at the target position
            output = output.clone()  # Clone to avoid in-place modification
            output[:, target_token_idx, :] = new_activation
            # output[batch_idx, seq_idx] = new_activation
            return output

        # Register the hook
        hook = self.model.layers[layer_idx].mlp.down_proj.register_forward_hook(hook_fn)
        """
        这个函数利用hook修改指定位置上的FFN层的激活值，并存储修改前的激活值。
        """
        return hook  # Return the hook handle for later removal if needed

    def generate_partitioning(self, vector, times=20):
        """
        生成一组按比例缩放的输入数据。

        参数:
            emb (torch.Tensor): 输入张量，形状为 (1, ffn_size)，表示一个基准向量。
            batch_size (int): 每个批次中的样本数量。
            num_batch (int): 批次的总数量。

        返回:
            partitioning (torch.Tensor): 形状为 (batch_size * num_batch, ffn_size) 的张量，包含生成的缩放输入。
            step (torch.Tensor): 每次增量的大小，形状为 (1, ffn_size)，表示每一步的变化量。
        equal partitioning

        """
        # vector = vector
        # vector = vector.unsqueeze(0)
        baseline = torch.zeros_like(vector)  # (1, ffn_size)

        # step: 计算每一步的增量，即 emb 和 baseline 之间的差距除以总的样本数。
        # 这是为了确保从 baseline 到 emb 的变化是均匀的，按比例分配给每个样本。
        step = (vector - baseline) / times  # (1, ffn_size) # 每个分量上的间隔值

        # res: 使用 baseline 和 step 生成 scaled input 的列表。
        # 通过 torch.add 和 step * i 逐步构建每个样本的值，并将这些值沿着第一个维度 (batch_size * num_batch) 拼接。
        partitioning = torch.cat(
            [torch.add(baseline, step * i) for i in range(times)], dim=0
        )  # (step, ffn_size)

        # 返回生成的输入数据和每一步的增量。
        return partitioning, step[0]

    # def generate_partitioning(self, vector, times=20):
    #     step = vector / times  # 直接计算步长
    #     return step * torch.arange(times, device=vector.device).unsqueeze(
    #         1
    #     ), step[0]  # 生成时避免存储中间矩阵

    def calulate_integrated_gradients(self, target_label):
        for i in range(self.config.num_hidden_layers):
            self._partitioning_activations[i].requires_grad_()
            prob = F.softmax(self._partitioning_logits[i], dim=1)
            o = torch.unbind(prob[:, target_label])
            (gradient,) = torch.autograd.grad(
                o, self._partitioning_activations[i], retain_graph=False
            )
            grad_summed = gradient.sum(dim=0)  # (ffn_size)
            integrated_gradients_of_layer = (
                grad_summed * self._partitioning_step[i]
            )  # (ffn_size)
            """
            在指定预测结果位置(MLM任务上最后预测单词)上,逐层计算integrated_gradients,integrated_gradients_of_layer意味着某层的ffn层逐神经元的integrated_gradients
            """
            self._integrated_gradients.append(integrated_gradients_of_layer)
        return self._integrated_gradients

    @property
    def get_intermediate_activations(self):
        """
        Returns the intermediate neuron activations from the last forward pass.
        Raises an error if intermediate_activations is empty.
        """
        if not self._intermediate_activations:
            raise ValueError(
                "Intermediate activations are empty. Ensure that a forward pass has been performed."
            )
        return self._intermediate_activations

    @property
    def original_activations(self):
        """
        Returns the original (before modification) intermediate neuron activations.
        """
        if not self._original_activations:
            raise ValueError(
                "Original activations are empty. Ensure that a forward pass has been performed."
            )
        return self._original_activations

    def clean(self):
        self._intermediate_activations.clear()
        self._original_activations.clear()
        self._partitioning_activations.clear()
        self._partitioning_step.clear()
        self._partitioning_last_hidden_state_last_pos.clear()
        self._partitioning_logits.clear()
        self._integrated_gradients.clear()
        self._args = None
        self._kwargs = None
        for hook in self._temporary_hooks:
            hook.remove()
        # torch.cuda.empty_cache()

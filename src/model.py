from transformers import BertForMaskedLM
import torch
import torch.nn.functional as F


def after_call(method_to_call):
    def decorator(func):
        def wrapper(self, *args, **kwargs):
            result = func(self, *args, **kwargs)
            getattr(self, method_to_call)()  # 调用指定方法
            return result

        return wrapper

    return decorator


class CustomBertForMaskedLM(BertForMaskedLM):
    def __init__(self, config):
        super().__init__(config)
        self._intermediate_activations = []  # 逐层的每个token在ffn层的输出
        self._original_activations = []
        self._mask_logits = None
        self._partitioning_activations = []
        self._partitioning_step = []
        self._partitioning_last_hidden_state = []
        self._PARTITIONING_TIMES = 20
        self._partitioning_logits = (
            []
        )  # self._partitioning_logits是last_hidden_state经过cls层的输出
        self._integrated_gradients = []
        self._temporary_hooks = []
        self._args = None
        self._kwargs = None
        for param in self.bert.parameters():
            param.requires_grad = False
        self.cls.requires_grad_(False)
        for layer in range(self.config.num_hidden_layers):
            intermediate_dense = self.bert.encoder.layer[layer].intermediate.dense
            output_dense = self.bert.encoder.layer[layer].output.dense
            intermediate_dense.weight.requires_grad = True
            # intermediate_dense.bias.requires_grad = True
            output_dense.weight.requires_grad = True
            # output_dense.bias.requires_grad = True

    def forward(self, *args, **kwargs):
        """
        保存现有参数以用于forward_with_partitioning
        """
        self._args = args
        self._kwargs = kwargs

        # Hook to capture intermediate activations
        def hook_fn(module, input, output):
            # 如果 _intermediate_activations 不为空，则清空它
            if len(self._intermediate_activations) == self.config.num_hidden_layers:
                self._intermediate_activations.clear()

            # 添加新的激活到 _intermediate_activations
            self._intermediate_activations.append(output)

        # Register hooks on intermediate layers
        hooks = []
        for layer in self.bert.encoder.layer:
            hooks.append(layer.intermediate.register_forward_hook(hook_fn))

        # Forward pass
        outputs = super().forward(*args, **kwargs)
        # Remove hooks after forward pass
        for hook in hooks:
            hook.remove()

        return outputs

    def forward_with_partitioning(self, target_position):
        """
        target_pos token idx
        """
        # 批量化处理提高性能
        batch_size = self._PARTITIONING_TIMES
        input_ids = self._kwargs["input_ids"].repeat(batch_size, 1)
        attention_mask = self._kwargs["attention_mask"].repeat(batch_size, 1)
        """
        找到对应token在每层的激活值，然后
        对每层的激活值进行间隔操作，并逐层记录
        """
        for layer in self._intermediate_activations:
            target_vector = layer[:, target_position, :]
            partitioning, step = self.generate_partitioning(target_vector)
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
                target_position,
                self._partitioning_activations[layer_idx],
            )
            self._temporary_hooks.append(hook)
            outputs = self.bert.forward(input_ids, attention_mask)
            # outputs = super().forward(input_ids, attention_mask)
            # outputs = super().forward(*self._args, **self._kwargs)
            self._partitioning_last_hidden_state.append(
                outputs.last_hidden_state
            )  # logits不可使用detach，否则会导致梯度丢失
            # self._partitioning_logits.append(
            #     outputs.logits
            # )  # logits不可使用detach，否则会导致梯度丢失

            self._partitioning_logits.append(
                self.cls(
                    self._partitioning_last_hidden_state[layer_idx][
                        :, target_position, :
                    ]
                )
            )  # 逐层地使用ffn激活值进行计算
        return self._partitioning_logits

    def modify_ffn_activation(self, layer_idx, target_position, new_activation):
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
            if new_activation.shape != output[:, target_position, :].shape:
                raise ValueError(
                    f"Shape mismatch: Expected {output[:,target_position,:].shape}, "
                    f"but got {new_activation.shape}."
                )

            # Modify the activation at the target position
            output = output.clone()  # Clone to avoid in-place modification
            output[:, target_position, :] = new_activation
            # output[batch_idx, seq_idx] = new_activation
            return output

        # Register the hook
        hook = self.bert.encoder.layer[layer_idx].intermediate.register_forward_hook(
            hook_fn
        )
        """
        这个函数利用hook修改指定位置上的FFN层的激活值，并存储修改前的激活值。
        """
        return hook  # Return the hook handle for later removal if needed

    def get_mask_logits(self, input_ids, attention_mask):
        """
        Returns logits at [MASK] positions for the given input.
        """
        self.eval()  # Ensure model is in eval mode
        with torch.no_grad():
            outputs = self.forward(
                input_ids=input_ids, attention_mask=attention_mask, return_dict=True
            )
            logits = outputs.logits

        # Find [MASK] token positions (token_id=103 for BERT by default)
        mask_token_id = 103
        # mask_token_id = self.config.mask_token_id
        mask_positions = input_ids == mask_token_id

        # Extract logits for [MASK] positions
        mask_logits = logits[mask_positions]

        return mask_logits

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

    def calulate_integrated_gradients(self, target_label):
        for i in range(self.config.num_hidden_layers):
            self._partitioning_activations[i].requires_grad_()
            prob = F.softmax(self._partitioning_logits[i], dim=1)
            o = torch.unbind(prob[:, target_label])
            (gradient,) = torch.autograd.grad(o, self._partitioning_activations[i])
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

    @property
    def mask_logits(self):
        """
        Returns the logits at [MASK] positions from the last forward pass.
        Raises an error if _mask_logits is None.
        """
        if self._mask_logits is None:
            raise ValueError(
                "Mask logits are not available. Ensure that a forward pass has been performed with [MASK] tokens."
            )
        return self._mask_logits

    def clean(self):
        self._intermediate_activations.clear()
        self._original_activations.clear()
        self._mask_logits = None
        self._partitioning_activations.clear()
        self._partitioning_step.clear()
        self._partitioning_last_hidden_state.clear()
        self._partitioning_logits.clear()
        self._integrated_gradients.clear()
        self._args = None
        self._kwargs = None
        for hook in self._temporary_hooks:
            hook.remove()

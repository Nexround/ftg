# 3.一个方法，接受指定神经元的激活值，并在模型中修改该神经元的激活值，在其他神经元激活值不变的情况下重新推理
from transformers import BertForMaskedLM
import torch
import torch.nn as nn


class CustomBertForMaskedLM(BertForMaskedLM):
    def __init__(self, config):
        super().__init__(config)
        self.intermediate_activations = []

    def forward(self, *args, **kwargs):
        # Hook to capture intermediate activations
        def hook_fn(module, input, output):
            self.intermediate_activations.append(output.detach().cpu())

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
        mask_token_id = self.config.mask_token_id
        mask_positions = input_ids == mask_token_id

        # Extract logits for [MASK] positions
        mask_logits = logits[mask_positions]

        return mask_logits

    @property
    def get_intermediate_activations(self):
        """
        Returns the intermediate neuron activations from the last forward pass.
        Raises an error if intermediate_activations is empty.
        """
        if not self.intermediate_activations:
            raise ValueError("Intermediate activations are empty. Ensure that a forward pass has been performed.")
        return self.intermediate_activations

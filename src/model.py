from transformers import BertForMaskedLM
import torch
import torch.nn as nn


class CustomBertForMaskedLM(BertForMaskedLM):
    def __init__(self, config):
        super().__init__(config)
        self._intermediate_activations = []
        self._mask_logits = None

    def forward(self, *args, **kwargs):
        # Hook to capture intermediate activations
        def hook_fn(module, input, output):
            self._intermediate_activations.append(output.detach().cpu())

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
        if not self._intermediate_activations:
            raise ValueError(
                "Intermediate activations are empty. Ensure that a forward pass has been performed."
            )
        return self._intermediate_activations

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

    def modify_ffn_activation(self, layer_idx, target_position, new_activation):
        """
        Modifies the hidden activations of a specific FFN layer at a specific position in the model.

        Args:
            layer_idx (int): Index of the transformer layer to modify (0-indexed).
            target_position (tuple): A tuple specifying the target position (batch_idx, seq_idx).
            new_activation (torch.Tensor): The new activation values with shape matching the FFN layer output
                                            (e.g., [intermediate_size]).
        """
        if not isinstance(new_activation, torch.Tensor):
            raise ValueError("new_activation must be a torch.Tensor.")

        def hook_fn(module, input, output):
            # Ensure the shape matches
            batch_idx, seq_idx = target_position
            if new_activation.shape != output[batch_idx, seq_idx].shape:
                raise ValueError(
                    f"Shape mismatch: Expected {output[batch_idx, seq_idx].shape}, "
                    f"but got {new_activation.shape}."
                )

            # Modify the activation at the target position
            output = output.clone()  # Clone to avoid in-place modification
            output[batch_idx, seq_idx] = new_activation
            return output

        # Register the hook
        hook = self.bert.encoder.layer[layer_idx].intermediate.register_forward_hook(
            hook_fn
        )

        return hook  # Return the hook handle for later removal if needed

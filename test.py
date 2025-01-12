from transformers import BertForMaskedLM
import torch
import torch.nn as nn


from transformers import BertForMaskedLM
import torch

from transformers import BertForMaskedLM
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import BertTokenizer, BertConfig
from transformers import BertForMaskedLM
import torch
import numpy as np

# Assuming CustomBertForMaskedLM is defined as per the provided code


def test_custom_bert_for_masked_lm():
    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load pre-trained BERT model and tokenizer
    model_name = "bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    config = BertConfig.from_pretrained(model_name)

    # Instantiate the model and move it to the appropriate device (CPU or GPU)
    model = CustomBertForMaskedLM(config).to(device)

    # Load pre-trained weights into the custom model
    model.bert.load_state_dict(
        BertForMaskedLM.from_pretrained(model_name).bert.state_dict()
    )

    # Tokenize some example input text with a [MASK] token
    text = "The quick brown fox jumps over the [MASK] dog."
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

    # Move input tensors to the same device as the model
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    # Perform the first forward pass to get original logits (before modification)
    original_outputs = model(
        input_ids=input_ids, attention_mask=attention_mask, return_dict=True
    )
    original_logits = original_outputs.logits

    # Get the original logits for the [MASK] position
    mask_token_id = 103
    mask_positions = input_ids == mask_token_id
    original_mask_logits = original_logits[mask_positions]

    # # Modify the activations of a specific FFN layer and position
    # target_position = (
    #     0,
    #     5,
    # )  # Example: batch_idx = 0, seq_idx = 5 (corresponding to the [MASK] token)
    # new_activation = torch.randn(model.config.intermediate_size).to(
    #     device
    # )  # Random tensor of appropriate size

    # # Modify the activation in the 3rd transformer layer
    # hook = model.modify_ffn_activation(
    #     layer_idx=2, target_position=target_position, new_activation=new_activation
    # )

    # Perform the second forward pass after modification
    # modified_outputs = model(
    #     input_ids=input_ids, attention_mask=attention_mask, return_dict=True
    # )
    # modified_logits = modified_outputs.logits

    # # Get the modified logits for the [MASK] position
    # modified_mask_logits = modified_logits[mask_positions]

    # Compare the logits before and after the modification
    # difference = torch.abs(original_mask_logits - modified_mask_logits)
    forward_with_partitioning = model.forward_with_partitioning(target_position=5)
    # Print the results
    print("Original logits for [MASK] position:", original_mask_logits)
    # print("Modified logits for [MASK] position:", modified_mask_logits)
    # print("Difference between original and modified logits:", difference)
    # print(f"Maximum difference: {difference.max()}")  # Print the maximum difference
    # partitioning, step = model.generate_partitioning(new_activation, times=20)
    re = model.calulate_integrated_gradients(target_label=123)
    # Optionally remove the hook after testing
    # hook.remove()
    from pprint import pprint

    pprint(model)


test_custom_bert_for_masked_lm()

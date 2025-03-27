from transformers import AutoModelForCausalLM, AutoTokenizer

import json
import torch
from src.loki.loki_qwen_config import LoKIQwen2Config
from src.loki.loki_qwen_model import LoKIQwen2ForCausalLM

target_neurons_path = "target_neurons/Qwen2.5-0.5B-Instruct/10.json"
save_dir = "/cache/models/custom-model-test"
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
with open(target_neurons_path, "r", encoding="utf-8") as f:
    target_neurons = json.load(f)

original_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype=torch.bfloat16
)

LoKIQwen2ForCausalLM.register_for_auto_class("AutoModelForCausalLM")
LoKIQwen2Config.register_for_auto_class()

loki_config = LoKIQwen2Config.from_pretrained(MODEL_NAME)
loki_config.target_neurons = target_neurons
loki_model = LoKIQwen2ForCausalLM.from_pretrained(
    pretrained_model_name_or_path=MODEL_NAME,
    config=loki_config,
    torch_dtype=torch.bfloat16,
)
loki_model.save_pretrained(save_dir)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.save_pretrained(save_dir)
original_model.save_pretrained(save_dir, is_main_process=False) # 仅保存模型参数

print("Done.")

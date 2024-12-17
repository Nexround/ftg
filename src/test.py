import torch
from transformers import BertModel, BertTokenizer
import json
import os

# 1. 加载 BERT 模型和 tokenizer
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)
config = model.config

# 2. 定义一个字典来存储激活值
activations = {}

# 3. 定义钩子函数来捕获中间层的激活值
def hook_fn(layer_name):
    def hook(module, input, output):
        # 将激活值按层编号和神经元编号保存
        activations[layer_name] = output.detach().cpu().numpy().tolist()
    return hook

# 4. 注册钩子：遍历所有 Transformer 层并注册钩子
hooks = []
for i, layer in enumerate(model.encoder.layer):
    # 注册钩子到每一层的 intermediate.dense（FFN部分）
    hook = layer.intermediate.dense.register_forward_hook(hook_fn(f"layer_{i}_ffn"))
    hooks.append(hook)

# 5. 输入一个文本样本并执行前向传播
input_text = "Hello, how are you?"
inputs = tokenizer(input_text, return_tensors="pt")
outputs = model(**inputs)

# 6. 保存激活值到硬盘（保存为 JSON 格式）
save_dir = "./activations"
os.makedirs(save_dir, exist_ok=True)

# 保存路径
activation_file_path = os.path.join(save_dir, "activations.json")

# 将激活值字典保存为 JSON 文件
with open(activation_file_path, 'w') as json_file:
    json.dump(activations, json_file)

print(f"Activations saved to {activation_file_path}")

# 7. 移除钩子
for hook in hooks:
    hook.remove()

import torch
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from shared import training_args, compute_metrics

# 指定需要微调的FFN神经元坐标列表
# trainable_neurons = [(0, 512), (1, 1024), (6, 2048), (11, 3071)]

import json

# 从 JSON 文件中读取数据
with open("/root/ftg/src/train/complement_1.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# 将数据转换为列表
trainable_neurons = list(data)
# 加载IMDB数据集
dataset = load_dataset("Yelp/yelp_review_full", cache_dir="/cache/huggingface/datasets")
tokenizer = BertTokenizer.from_pretrained(
    "bert-base-uncased", cache_dir="/cache/huggingface/hub"
)


# 数据预处理
def tokenize_function(examples):
    return tokenizer(
        examples["text"], padding="max_length", truncation=True, max_length=512
    )


tokenized_train = dataset["train"].map(tokenize_function, batched=True)
tokenized_test = dataset["test"].map(tokenize_function, batched=True)

# 4. 设置 PyTorch Dataset
tokenized_train.set_format(
    type="torch", columns=["input_ids", "attention_mask", "label"]
)
tokenized_test.set_format(
    type="torch", columns=["input_ids", "attention_mask", "label"]
)

# 加载模型
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased", num_labels=10, cache_dir="/cache/huggingface/hub"
)

# 冻结所有参数
for param in model.bert.parameters():
    param.requires_grad = False


def unfreeze_ffn_connections_with_hooks_optimized(model, trainable_neurons):
    hooks = []

    # 按层分组 trainable_neurons，减少 hook 数量
    from collections import defaultdict
    layer_to_neurons = defaultdict(list)
    for layer, neuron_index in trainable_neurons:
        layer_to_neurons[layer].append(neuron_index)

    for layer, neuron_indices in layer_to_neurons.items():
        # 获取当前层的中间层和输出层
        intermediate_dense = model.bert.encoder.layer[layer].intermediate.dense
        output_dense = model.bert.encoder.layer[layer].output.dense

        # 确保权重和偏置的 requires_grad 为 True
        intermediate_dense.weight.requires_grad = True
        intermediate_dense.bias.requires_grad = True
        output_dense.weight.requires_grad = True
        output_dense.bias.requires_grad = True

        # 注册单个钩子函数，针对输入权重的所有指定神经元
        def input_weight_hook(grad):
            mask = torch.zeros_like(grad)
            mask[neuron_indices, :] = 1  # 只允许指定神经元的梯度
            return grad * mask

        hooks.append(intermediate_dense.weight.register_hook(input_weight_hook))

        # 注册单个钩子函数，针对输出权重的所有指定神经元
        def output_weight_hook(grad):
            mask = torch.zeros_like(grad)
            mask[:, neuron_indices] = 1  # 只允许指定神经元的梯度
            return grad * mask

        hooks.append(output_dense.weight.register_hook(output_weight_hook))

        # 注册单个钩子函数，针对输入偏置的所有指定神经元
        def input_bias_hook(grad):
            mask = torch.zeros_like(grad)
            mask[neuron_indices] = 1  # 只允许指定神经元的梯度
            return grad * mask

        hooks.append(intermediate_dense.bias.register_hook(input_bias_hook))

        # 输出层偏置不需要区分神经元，保持全梯度
        def output_bias_hook(grad):
            return grad  # 不修改偏置梯度

        hooks.append(output_dense.bias.register_hook(output_bias_hook))

    return hooks  # 返回 hooks 以便后续管理和清理

hooks = unfreeze_ffn_connections_with_hooks_optimized(model, trainable_neurons)
model.classifier.weight.requires_grad = True
# # 分类头的参数允许训练
# for param in model.classifier.parameters():
#     param.requires_grad = True
for param in model.parameters():
    if param.requires_grad:
        print(param.numel())
for name, param in model.named_parameters():
    if param.requires_grad:
        print(f"Parameter: {name}, Shape: {param.shape}")
trainable_param_count = sum(
    param.numel() for param in model.parameters() if param.requires_grad
)
print(f"Total Trainable Parameters: {trainable_param_count}")
print(model)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    processing_class=tokenizer,
    compute_metrics=compute_metrics,
)

# 开始训练
trainer.train()
eval_results = trainer.evaluate()
print(f"Evaluation results: {eval_results}")

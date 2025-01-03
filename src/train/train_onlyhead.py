import torch
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# 指定需要微调的FFN神经元坐标列表
# trainable_neurons = [(0, 512), (1, 1024), (6, 2048), (11, 3071)]


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary"
    )
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}


# 加载IMDB数据集
dataset = load_dataset("imdb", cache_dir="/cache/huggingface/datasets")
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
    "bert-base-uncased", num_labels=2, cache_dir="/cache/huggingface/hub"
)

# 冻结所有参数
for param in model.bert.parameters():
    param.requires_grad = False


# # 解冻指定FFN神经元的权重
# def get_ffn_weights(layer, neuron_index):
#     return (
#         model.bert.encoder.layer[layer].intermediate.dense.weight[neuron_index],
#         model.bert.encoder.layer[layer].intermediate.dense.bias[neuron_index],
#     )


# for layer, neuron_index in trainable_neurons:
#     weight, bias = get_ffn_weights(layer, neuron_index)
#     weight.requires_grad = True
#     bias.requires_grad = True

# 分类头的参数允许训练
for param in model.classifier.parameters():
    param.requires_grad = True

# 设置训练参数
training_args = TrainingArguments(
    output_dir="./results",  # 保存模型的路径
    evaluation_strategy="epoch",  # 每个 epoch 进行一次评估
    learning_rate=5e-4,  # 学习率
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,  # 训练 epoch 数
    weight_decay=0.01,  # 权重衰减
    logging_dir="./logs",  # 日志路径
    logging_steps=500,
    save_steps=1000,
    save_total_limit=2,  # 最多保存两个模型
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# 开始训练
trainer.train()
eval_results = trainer.evaluate()
print(f"Evaluation results: {eval_results}")

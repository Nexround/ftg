import time
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

# 加载IMDB数据集
dataset = load_dataset("fancyzhx/yelp_polarity", cache_dir="/cache/huggingface/datasets")
tokenizer = BertTokenizer.from_pretrained(
    "bert-base-uncased", cache_dir="/cache/huggingface/hub"
)


# 数据预处理
def tokenize_function(examples):
    return tokenizer(
        examples["text"], padding="max_length", truncation=True, max_length=512
    )


tokenized_train = dataset["train"].map(tokenize_function, batched=True, num_proc=16)
tokenized_test = dataset["test"].map(tokenize_function, batched=True, num_proc=16)

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
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# 开始训练
trainer.train()
eval_results = trainer.evaluate()
print(f"Evaluation results: {eval_results}")

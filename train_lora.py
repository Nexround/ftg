from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from datasets import load_dataset, ClassLabel
from peft import LoraConfig, get_peft_model
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from src.module.func import compute_metrics  # 加载IMDB数据集
import time
import os

dataset = load_dataset("imdb", cache_dir="/cache/huggingface/datasets")

# 加载BERT分词器和模型
tokenizer = BertTokenizer.from_pretrained(
    "bert-base-uncased", cache_dir="/cache/huggingface/hub"
)
model = BertForSequenceClassification.from_pretrained(
    "/root/ftg/results/new_6tags_agnews_based_imdb", cache_dir="/cache/huggingface/hub"
)

new_class_labels = ClassLabel(
        num_classes=6, names=["World", "Sports", "Business", "Sci/Tech", "neg", "pos"]
    )
dataset = dataset.cast_column("label", new_class_labels)

    # 应用映射函数
    # updated_dataset = dataset.map(update_label)
def update_label(example):
    original_label = example["label"]

    # 重新映射原始标签d
    if original_label == 0:
        example["label"] = 4  # 原标签 0 -> 新标签 3
    elif original_label == 1:
        example["label"] = 5  # 原标签 1 -> 新标签 4

    return example

dataset = dataset.map(update_label)
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
# LoRA配置
lora_config = LoraConfig(
    r=8, lora_alpha=32, target_modules=["query", "key"], lora_dropout=0.1, bias="none"
)

# 使用LoRA调整模型
model = get_peft_model(model, lora_config)

trainable_param_count = sum(
    param.numel() for param in model.parameters() if param.requires_grad
)
print(f"Total Trainable Parameters: {trainable_param_count}")
print(model)
log_dir = f"logs/run_{time.strftime('%Y%m%d-%H%M%S')}_{os.path.basename(__file__)}"
training_args = TrainingArguments(
    output_dir="./results",  # 保存模型的路径
    eval_strategy="epoch",  # 每个 epoch 进行一次评估
    learning_rate=5e-5,  # 学习率
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,  # 训练 epoch 数
    weight_decay=0.01,  # 权重衰减
    logging_dir=log_dir,  # 日志路径
    logging_steps=500,
    save_steps=1000,
    save_total_limit=2,  # 最多保存两个模型
)
# 定义Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# 训练
trainer.train()
eval_results = trainer.evaluate()
print(f"Evaluation results: {eval_results}")
# 保存模型
trainer.save_model("./lora_bert_imdb")

from datasets import load_dataset
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# 1. 加载数据集
dataset = load_dataset("imdb", cache_dir="/cache/huggingface/datasets")

# 2. 加载 BERT 分词器
tokenizer = BertTokenizer.from_pretrained(
    "bert-base-cased", cache_dir="/cache/huggingface/hub"
)


# 3. 数据预处理
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

# 5. 加载预训练的 BERT 模型
model = BertForSequenceClassification.from_pretrained(
    "bert-base-cased", cache_dir="/cache/huggingface/hub", num_labels=2
)


# 6. 定义评估函数
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary"
    )
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}


# 7. 设置训练参数
training_args = TrainingArguments(
    output_dir="./results",  # 保存模型的路径
    evaluation_strategy="epoch",  # 每个 epoch 进行一次评估
    learning_rate=5e-5,  # 学习率
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,  # 训练 epoch 数
    weight_decay=0.01,  # 权重衰减
    logging_dir="./logs",  # 日志路径
    logging_steps=500,
    save_steps=1000,
    save_total_limit=2,  # 最多保存两个模型
)

# 8. 定义 Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# 9. 开始训练
trainer.train()

# 10. 模型评估
eval_results = trainer.evaluate()
print(f"Evaluation results: {eval_results}")

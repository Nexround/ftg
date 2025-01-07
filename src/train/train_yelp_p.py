from datasets import load_dataset
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from shared import compute_metrics
import time
import os
# 1. 加载数据集
dataset = load_dataset("fancyzhx/yelp_polarity", cache_dir="/cache/huggingface/datasets")

# 2. 加载 BERT 分词器
tokenizer = BertTokenizer.from_pretrained(
    "bert-base-uncased", cache_dir="/cache/huggingface/hub"
)
print(f"Current file name: {os.path.basename(__file__)}")

# 3. 数据预处理
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

# 5. 加载预训练的 BERT 模型
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased", cache_dir="/cache/huggingface/hub", num_labels=2
)
trainable_param_count = sum(
    param.numel() for param in model.parameters() if param.requires_grad
)
print(f"Total Trainable Parameters: {trainable_param_count}")
print(model)
log_dir = f"logs/run_{time.strftime('%Y%m%d-%H%M%S')}_{os.path.basename(__file__)}"
training_args = TrainingArguments(
    output_dir="./results",  # 保存模型的路径
    evaluation_strategy="epoch",  # 每个 epoch 进行一次评估
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

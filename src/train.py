from transformers import BertTokenizer, BertForMaskedLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset

# 1. 加载数据集（可以替换为自己的数据集）
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")  # 使用 WikiText 数据集作为示例

# 2. 加载预训练的BERT tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# 3. 数据预处理：将文本数据编码为 BERT 输入格式
def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

tokenized_datasets = dataset.map(preprocess_function, batched=True, remove_columns=["text"])

# 4. 加载预训练的BERT模型（用于掩码语言模型）
model = BertForMaskedLM.from_pretrained("bert-base-uncased")

# 5. 定义数据整理器，用于动态掩码
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,           # 启用 MLM
    mlm_probability=0.15  # 掩码的概率
)

# 6. 定义训练参数
training_args = TrainingArguments(
    output_dir="./results_mlm",       # 保存模型的目录
    evaluation_strategy="epoch",     # 每个 epoch 后评估
    learning_rate=5e-5,              # 学习率
    per_device_train_batch_size=128,  # 每个设备的训练批量大小
    per_device_eval_batch_size=128,   # 每个设备的评估批量大小
    num_train_epochs=3,              # 训练轮数
    weight_decay=0.01,               # 权重衰减
    save_steps=500,                  # 保存检查点的步数
    save_total_limit=2,              # 保留的最多检查点数量
    logging_dir="./logs",            # 日志存储路径
    logging_steps=10,                # 日志记录步数
    # load_best_model_at_end=True      # 在训练结束时加载最佳模型
)
# training_args = TrainingArguments(
#     output_dir="./results",                   # 模型保存路径
#     per_device_train_batch_size=16,           # 每个设备的训练批次大小
#     per_device_eval_batch_size=16,            # 每个设备的评估批次大小
#     num_train_epochs=3,                       # 训练的轮数
#     evaluation_strategy="epoch",              # 每个 epoch 后评估一次
#     save_strategy="epoch",                    # 每个 epoch 后保存一次模型
#     logging_dir="./logs",                     # 日志路径
#     load_best_model_at_end=True,              # 在训练结束时加载最佳模型
#     metric_for_best_model="accuracy",        # 根据哪种指标选择最佳模型（根据需求选择）
#     save_total_limit=2,                       # 保留的模型检查点数量
# )
# 7. 创建 Trainer
trainer = Trainer(
    model=model,                             # 模型
    args=training_args,                      # 训练参数
    train_dataset=tokenized_datasets["train"],  # 训练集
    eval_dataset=tokenized_datasets["validation"],  # 验证集
    data_collator=data_collator,             # 数据整理器
    tokenizer=tokenizer                      # Tokenizer
)

# 8. 训练模型
trainer.train()

# 9. 保存微调的模型
trainer.save_model("./fine_tuned_bert_mlm")

print("继续训练完成！模型已保存至 ./fine_tuned_bert_mlm")

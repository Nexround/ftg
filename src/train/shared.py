import time
from transformers import (
    TrainingArguments,
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import os


log_dir = f"logs/run_{time.strftime('%Y%m%d-%H%M%S')}_{os.path.basename(__file__)}"
training_args = TrainingArguments(
    output_dir="./results",  # 保存模型的路径
    eval_strategy="epoch",  # 每个 epoch 进行一次评估
    learning_rate=5e-4,  # 学习率
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,  # 训练 epoch 数
    weight_decay=0.01,  # 权重衰减
    logging_dir=log_dir,  # 日志路径
    logging_steps=500,
    save_steps=1000,
    save_total_limit=2,  # 最多保存两个模型
    save_strategy="epoch",
)

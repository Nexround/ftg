import time
from transformers import (
    TrainingArguments,
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import os

def compute_metrics(pred):
    """
    计算模型的评价指标，包括准确率、精确率、召回率和F1分数。

    Args:
        pred: 一个包含两个字段的对象，分别是`predictions`和`label_ids`。
              `predictions`是模型的预测结果，`label_ids`是真实标签。

    Returns:
        Dict[str, float]: 各种评价指标的字典。
    """
    # 获取预测结果和真实标签
    predictions = pred.predictions.argmax(axis=-1)  # 对每个样本选择概率最高的类别
    label_ids = pred.label_ids

    # 计算评价指标
    accuracy = accuracy_score(label_ids, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        label_ids, predictions, average="weighted"  # 加权平均，适合多分类
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
def compute_metrics_b(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary"
    )
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}


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

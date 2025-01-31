from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from src.module.func import (
    compute_metrics,
)
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
from tqdm import tqdm
from datasets import ClassLabel

# 加载数据集（以IMDb影评数据集为例）
# dataset = load_dataset("imdb", cache_dir="/cache/huggingface/datasets", split="test")
# dataset = load_from_disk("/cache/huggingface/datasets/SciNews_3labels")["test"]
dataset = load_from_disk("/cache/huggingface/datasets/ag_news_14labels")['test']
# dataset = load_dataset(
#     "/cache/huggingface/datasets/ag_news_14labels", cache_dir="/cache/huggingface/datasets", split="test"
# )
# target_labels = ["World", "Sports", "Business"]  # 例：指定要保留的标签
# target_labels = [0, 1, 2]  # 例：指定要保留的标签


# dataset = dataset.filter(lambda x: x["label"] in target_labels)
# def update_label(example):
#     original_label = example["label"]

#     # 重新映射原始标签d
#     if original_label == "Physics":
#         example["label"] = 3  # 原标签 0 -> 新标签 3
#     elif original_label == "Medicine":
#         example["label"] = 4  # 原标签 1 -> 新标签 4
#     elif original_label == "Biology":
#         example["label"] = 5  # 原标签 1 -> 新标签 4

#     return example


# dataset = dataset.map(update_label)
# 加载第一个模型
# fine_tuned_bert_model = AutoModelForSequenceClassification.from_pretrained(
#     "/openbayes/home/ftg/results/train_full_imdb_on_agnews/checkpoint-4689"
# )
print(dataset)
fine_tuned_bert_model = AutoModelForSequenceClassification.from_pretrained(
    "/root/ftg/results/full__cache_huggingface_datasets_ag_news_14labels_01_21_04:11/checkpoint-206087"
)
# fine_tuned_bert_model = AutoModelForSequenceClassification.from_pretrained(
#     "/openbayes/home/ftg/results/agnews_checkpoint-22500"
# )
# fine_tuned_bert_model = AutoModelForSequenceClassification.from_pretrained(
#     "/openbayes/home/ftg/results/train_target_token_cls_agnews_bert_imdb_cls_complement_1_high"
# )

# 加载第二个模型
# fine_tuned_classifier_model = AutoModelForSequenceClassification.from_pretrained(
#     "/openbayes/home/ftg/results/agnews_checkpoint-22500"
# )
# fine_tuned_classifier_model = AutoModelForSequenceClassification.from_pretrained(
#     "/openbayes/home/ftg/results/train_target_token_cls_agnews_bert_imdb_cls_complement_1_high"
# )

# fine_tuned_bert_model.classifier = fine_tuned_classifier_model.classifier
# del fine_tuned_classifier_model
# 加载微调后的BERT模型
# model_name = "your-finetuned-bert-model"  # 替换为实际的模型路径或名称
# model = AutoModelForSequenceClassification.from_pretrained("/openbayes/home/ftg/results/train_full_imdb")
model = fine_tuned_bert_model
# model = fine_tuned_bert_model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


tokenizer = AutoTokenizer.from_pretrained(
    "bert-base-cased", cache_dir="/cache/huggingface/hub"
)
model.eval()  # 设置为评估模式


# def update_label(example):
#     original_label = example["label"]

#     # 重新映射原始标签
#     if original_label == 0:
#         example["label"] = 4  # 原标签 0 -> 新标签 3
#     elif original_label == 1:
#         example["label"] = 5  # 原标签 1 -> 新标签 4

#     return example


# new_class_labels = ClassLabel(
#     num_classes=6, names=["World", "Sports", "Business", "Sci/Tech", "neg", "pos"]
# )
# dataset = dataset.cast_column("label", new_class_labels)
# # 应用映射函数
# dataset = dataset.map(update_label)


# 数据预处理
def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, padding=True, max_length=512)


encoded_test_dataset = dataset.map(preprocess_function, batched=True, num_proc=40)
encoded_test_dataset = encoded_test_dataset.with_format(
    type="torch", columns=["input_ids", "attention_mask", "label"], device=device
)
encoded_test_dataset = encoded_test_dataset.batch(1)


# 计算准确率
def compute_accuracy(predictions, references):
    return (np.array(predictions) == np.array(references)).mean()


# 推理并评估
all_predictions = []
all_labels = []

with torch.no_grad():
    for batch in tqdm(encoded_test_dataset):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["label"]
        try:
            outputs = model(
                input_ids=input_ids, attention_mask=attention_mask
            )  # .item() 将一个包含单个值的张量转换为一个 Python 原生数值类型。input_ids=*batch["input_ids"], attention_mask=batch["attention_mask"]
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)

            all_predictions.append(predictions.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
        except:
            continue

# 计算准确率
accuracy = compute_accuracy(all_predictions, all_labels)
print(f"Accuracy: {accuracy:.4f}")

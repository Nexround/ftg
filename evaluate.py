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
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
from tqdm import tqdm

# 加载数据集（以IMDb影评数据集为例）
dataset = load_dataset("imdb", cache_dir="/cache/huggingface/datasets")
# dataset = load_dataset("fancyzhx/ag_news", cache_dir="/cache/huggingface/datasets")
# 加载第一个模型
# fine_tuned_bert_model = AutoModelForSequenceClassification.from_pretrained(
#     "/openbayes/home/ftg/results/train_full_imdb_on_agnews/checkpoint-4689"
# )
fine_tuned_bert_model = AutoModelForSequenceClassification.from_pretrained(
    "/openbayes/home/ftg/results/agnews_checkpoint-22500"
)
# fine_tuned_bert_model = AutoModelForSequenceClassification.from_pretrained(
#     "/openbayes/home/ftg/results/train_target_token_cls_agnews_bert_imdb_cls_complement_1_high"
# )

# 加载第二个模型
# fine_tuned_classifier_model = AutoModelForSequenceClassification.from_pretrained(
#     "/openbayes/home/ftg/results/agnews_checkpoint-22500"
# )
fine_tuned_classifier_model = AutoModelForSequenceClassification.from_pretrained(
    "/openbayes/home/ftg/results/train_target_token_cls_agnews_bert_imdb_cls_complement_1_high"
)

fine_tuned_bert_model.classifier = fine_tuned_classifier_model.classifier
# del fine_tuned_classifier_model
# 加载微调后的BERT模型
# model_name = "your-finetuned-bert-model"  # 替换为实际的模型路径或名称
# model = AutoModelForSequenceClassification.from_pretrained("/openbayes/home/ftg/results/train_full_imdb")
model = fine_tuned_bert_model
# model = fine_tuned_bert_model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
model.eval()  # 设置为评估模式


# 数据预处理
def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, padding=True, max_length=512)


encoded_test_dataset = dataset["test"].map(preprocess_function, batched=True)
encoded_test_dataset.set_format(
    type="torch", columns=["input_ids", "attention_mask", "label"]
)


# 计算准确率
def compute_accuracy(predictions, references):
    return (np.array(predictions) == np.array(references)).mean()


# 推理并评估
all_predictions = []
all_labels = []

with torch.no_grad():
    for batch in tqdm(encoded_test_dataset):
        input_ids = (
            batch["input_ids"]
            .unsqueeze(0)
            .to("cuda" if torch.cuda.is_available() else "cpu")
        )
        attention_mask = (
            batch["attention_mask"]
            .unsqueeze(0)
            .to("cuda" if torch.cuda.is_available() else "cpu")
        )
        labels = batch["label"]

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1).item()

        all_predictions.append(predictions)
        all_labels.append(labels)

# 计算准确率
accuracy = compute_accuracy(all_predictions, all_labels)
print(f"Accuracy: {accuracy:.4f}")

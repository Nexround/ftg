import time
import argparse
import json
from functools import partial
from transformers import (
    Trainer,
    TrainingArguments,
    AutoTokenizer,
    AutoModelForSequenceClassification,
)
from datasets import load_dataset

from src.module.func import (
    unfreeze_ffn_connections_with_hooks_optimized,
    compute_metrics,
    parse_comma_separated,
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset", default=None, type=parse_comma_separated, required=True
    )
    parser.add_argument("--target_neurons_path", default=None, type=str)

    parser.add_argument(
        "--model",
        default=None,
        type=str,
        required=True,
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
    )
    parser.add_argument(
        "--output_prefix",
        default=None,
        type=str,
        required=True,
        help="The output prefix to indentify each running of experiment. ",
    )

    parser.add_argument("--learning_rate", default=5e-5, type=float)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--num_labels", default=2, type=int)
    parser.add_argument("--num_train_epochs", default=3, type=int)
    parser.add_argument(
        "--lora", default=False, action="store_true", help="Use LoRA optimizer"
    )
    parser.add_argument("--train_target_neurons", default=False, action="store_true")
    parser.add_argument("--only_head", default=False, action="store_true")
    parser.add_argument("--full", default=False, action="store_true")
    args = parser.parse_args()

    dataset = load_dataset(*args.dataset, cache_dir="/cache/huggingface/datasets")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, cache_dir="/cache/huggingface/hub"
    )
    # 加载模型
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=args.num_labels,
        cache_dir="/cache/huggingface/hub",
        ignore_mismatched_sizes=True,
        # problem_type="multi_label_classification", # 单文本多标签
    )

    # 数据预处理
    def tokenize_function(examples):
        return tokenizer(
            examples["text"], padding="max_length", truncation=True, max_length=512
        )

    # dataset = dataset.rename_column("class_index", "label")

    tokenized_train = (
        dataset["train"]
        .shuffle(seed=42)
        .map(tokenize_function, batched=True, num_proc=32)
    )
    tokenized_test = (
        dataset["test"]
        .shuffle(seed=42)
        .map(tokenize_function, batched=True, num_proc=32)
    )

    # 4. 设置 PyTorch Dataset
    tokenized_train.set_format(
        type="torch", columns=["input_ids", "attention_mask", "label"]
    )
    tokenized_test.set_format(
        type="torch", columns=["input_ids", "attention_mask", "label"]
    )

    if args.only_head:
        for param in model.bert.parameters():
            param.requires_grad = False
        for param in model.classifier.parameters():
            param.requires_grad = True
        print("=== Training only_head ===")
    elif args.train_target_neurons:
        for param in model.bert.parameters():
            param.requires_grad = False
        for param in model.classifier.parameters():
            param.requires_grad = True
        with open(args.target_neurons_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        trainable_neurons = list(data)
        hooks = unfreeze_ffn_connections_with_hooks_optimized(model, trainable_neurons)

        print("=== Training train_target_neurons ===")
    elif args.full:
        print("=== Training full ===")

    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"Parameter: {name}, Shape: {param.shape}")
    trainable_param_count = sum(
        param.numel() for param in model.parameters() if param.requires_grad
    )
    print(f"Total Trainable Parameters: {trainable_param_count}")
    print(model)
    log_dir = f"logs/run_{time.strftime('%Y_%m_%d_%H:%M:%S')}_{args.dataset}"
    training_args = TrainingArguments(
        output_dir=f"{args.output_dir}/run_{time.strftime('%m_%d_%H:%M')}_{args.dataset}",  # 保存模型的路径
        eval_strategy="epoch",  # 每个 epoch 进行一次评估
        learning_rate=args.learning_rate,  # 学习率
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.num_train_epochs,  # 训练 epoch 数
        # weight_decay=0.01,  # 权重衰减
        logging_dir=log_dir,  # 日志路径
        logging_steps=500,
        save_steps=1000,
        save_total_limit=2,  # 最多保存两个模型
        save_strategy="epoch",
        fp16=True,
        # label_names=["class_index"]
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        processing_class=tokenizer,
        compute_metrics=(
            partial(compute_metrics, method="binary")
            if args.num_labels == 2
            else compute_metrics
        ),
    )

    # 开始训练
    trainer.train()
    eval_results = trainer.evaluate()
    print(f"Evaluation results: {eval_results}")

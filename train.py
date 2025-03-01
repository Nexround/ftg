import time
import argparse
import json
from functools import partial
from transformers import (
    Trainer,
    TrainingArguments,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoConfig,
)
from peft import LoraConfig, get_peft_model

from datasets import load_dataset
import torch
from src.module.func import (
    unfreeze_ffn_connections_with_hooks_optimized,
    compute_metrics,
    parse_comma_separated,
    generate_minimal_dot_product_vectors,
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
    parser.add_argument("--lora", default=False, action="store_true")
    parser.add_argument("--train_target_neurons", default=False, action="store_true")
    parser.add_argument("--only_head", default=False, action="store_true")
    parser.add_argument("--full", default=False, action="store_true")
    args = parser.parse_args()

    dataset = load_dataset(*args.dataset, cache_dir="/cache/huggingface/datasets")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, cache_dir="/cache/huggingface/hub"
    )
    # 加载模型
    config = AutoConfig.from_pretrained(args.model)

    # 修改 id2label 和 label2id
    config.label2id = {
        "World": 0,
        "Sports": 1,
        "Business": 2,
        "Sci/Tech": 3,
        "neg": 4,
        "pos": 5,
    }
    config.id2label = {value: key for key, value in config.label2id.items()}
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        config=config,
        # num_labels=args.num_labels,
        cache_dir="/cache/huggingface/hub",
        # ignore_mismatched_sizes=True,
        # problem_type="multi_label_classification", # 单文本多标签
    )

    # 数据预处理
    def tokenize_function(examples):
        return tokenizer(
            examples["text"], padding="max_length", truncation=True, max_length=512
        )

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
        classifier_weights = generate_minimal_dot_product_vectors(
            dim=768, num_vectors=6
        )
        model.classifier.weight.data = torch.tensor(
            classifier_weights, dtype=torch.float32
        )
        model.classifier.bias.data.fill_(0.0)
        for param in model.classifier.parameters():
            param.requires_grad = False
    elif args.lora:
        # LoRA配置
        lora_config = LoraConfig(
            r=8,
            lora_alpha=32,
            target_modules=["query", "key"],
            lora_dropout=0.1,
            bias="none",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"Parameter: {name}, Shape: {param.shape}")
    trainable_param_count = sum(
        param.numel() for param in model.parameters() if param.requires_grad
    )
    print(f"Total Trainable Parameters: {trainable_param_count}")
    print(model)

    log_dir = f"logs/run_{time.strftime('%Y_%m_%d_%H:%M:%S')}_{args.dataset}"
    f_dataset = args.dataset.replace("/", "_")
    output_dir = f"{args.output_dir}/{f_dataset}_{time.strftime('%m_%d_%H:%M')}"

    training_args = TrainingArguments(
        output_dir=output_dir,  # 保存模型的路径
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
    with open(f"{output_dir}/training_args.json", "w", encoding="utf-8") as f:
        json.dump(training_args.to_dict(), f, indent=4)
    eval_results = trainer.evaluate()
    with open(f"{output_dir}/eval_results.json", "w", encoding="utf-8") as f:
        json.dump(eval_results, f, indent=4)

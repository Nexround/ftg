import time
import argparse
import json
from enum import Enum


from transformers import (
    Trainer,
    TrainingArguments,
    AutoTokenizer,
    AutoModelForCausalLM,
)
from peft import LoraConfig, get_peft_model

from datasets import load_dataset
from src.module.func import (
    parse_comma_separated,
)

import torch
from torch.optim import AdamW
from src.module.loki_linear import replace_all_target_linear_qwen


def data_collator_t(batch):
    # 转换消息格式
    batch_converted_messages = []
    for conv in batch:
        converted = []
        for msg in conv["conversations"]:
            role = msg["from"]
            content = msg["value"]
            if role == "system":
                converted.append({"role": "system", "content": content})
            elif role == "human":
                converted.append({"role": "user", "content": content})
            elif role == "gpt":
                converted.append({"role": "assistant", "content": content})
        batch_converted_messages.append(converted)

    # 生成格式化文本
    texts = [
        tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        for messages in batch_converted_messages
    ]

    # 批量编码文本
    model_inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        return_tensors="pt",
        return_attention_mask=True,
    )
    input_ids = model_inputs["input_ids"]
    attention_mask = model_inputs["attention_mask"]
    batch_size, max_seq_len = input_ids.shape

    # 生成labels
    all_labels = []
    for i in range(batch_size):
        # 获取当前样本的最后一条assistant消息
        messages = batch_converted_messages[i]
        last_msg = messages[-1]
        assert last_msg["role"] == "assistant", "Last message must be from assistant"

        # 生成labels文本并编码
        assistant_content = last_msg["content"]
        labels_text = assistant_content + tokenizer.eos_token + "\n"
        labels_ids = tokenizer(
            labels_text,
            add_special_tokens=False,
            truncation=True,
            max_length=max_seq_len,  # 防止溢出
        ).input_ids

        # 计算有效内容长度（排除padding）
        original_length = attention_mask[i].sum().item()

        # 确定标签起始位置
        start_pos = original_length - len(labels_ids)
        if start_pos < 0:
            # 截断过长的labels_ids
            labels_ids = labels_ids[-original_length:]
            start_pos = 0

        # 创建并填充labels张量
        labels = torch.full((max_seq_len,), -100, dtype=torch.long)
        end_pos = start_pos + len(labels_ids)
        labels[start_pos:end_pos] = torch.tensor(labels_ids)
        all_labels.append(labels)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": torch.stack(all_labels),
    }


class TrainOption(Enum):
    LORA = "lora"
    FFN = "ffn"
    FULL = "full"
    C_FULL = "c_full"  # continue full
    TARGET = "loki"
    ONLY_HEAD = "only_head"


if __name__ == "__main__":

    print("***** CUDA.empty_cache() *****")
    torch.cuda.empty_cache()
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
        "--label_json",
        default=None,
        type=str,
        required=False,
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
    )
    parser.add_argument(
        "--ds_config",
        default=None,
        type=str,
        required=False,
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
    parser.add_argument("--num_train_epochs", default=3, type=int)
    parser.add_argument("--lora_rank", default=8, type=int)
    parser.add_argument("--lora_alpha", default=32, type=int)

    parser.add_argument(
        "--train_option",
        choices=[e.value for e in TrainOption],  # 使用枚举的值作为有效的参数选项
        required=True,
        help="Choose an option: 'option1', 'option2', or 'option3'",
    )

    args = parser.parse_args()

    dataset = load_dataset(*args.dataset, cache_dir="/cache/huggingface/datasets")
    dataset = dataset.select_columns(["conversations"])
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, cache_dir="/cache/huggingface/hub"
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        cache_dir="/cache/huggingface/hub",
    )

    dataset = dataset["train"].shuffle(seed=42)

    print(args.train_option)
    if args.train_option == "only_head":
        print("====================")
        print("=== Training only_head ===")
        print("====================")

        for param in model.bert.parameters():
            param.requires_grad = False
        for param in model.classifier.parameters():
            param.requires_grad = True

    elif args.train_option == "loki":
        print("====================")
        print("=== Training loki ===")
        print("====================")

        for param in model.model.parameters():
            param.requires_grad = False
        for param in model.lm_head.parameters():
            param.requires_grad = False
        with open(args.target_neurons_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        trainable_neurons = list(data)
        # print(f"trainable_neurons: {x:= }")
        replace_all_target_linear_qwen(model, trainable_neurons)
        print(model)

    elif args.train_option == "ffn":
        print("====================")
        print("=== Training ffn ===")
        print("====================")

        for param in model.model.parameters():
            param.requires_grad = False
        for param in model.lm_head.parameters():
            param.requires_grad = False
        for layer in model.model.layers:
            layer.mlp.down_proj.weight.requires_grad = True

    elif args.train_option == "full":
        print("=====================")
        print("=== Training full ===")
        print("=====================")

    elif args.train_option == "c_full":
        print("=====================")
        print("=== Continue Training full ===")
        print("=====================")

    elif args.train_option == "lora":
        print("=====================")
        print("=== Training lora ===")
        print("=====================")
        # LoRA配置
        lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            target_modules=["down_proj"],
            lora_dropout=0.1,
            bias="none",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        print("=== Training lora ===")

    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"Parameter: {name}, Shape: {param.shape}")
    trainable_param_count = sum(
        param.numel() for param in model.parameters() if param.requires_grad
    )
    print(f"Total Trainable Parameters: {trainable_param_count}")
    # print(model)
    log_dir = f"logs/run_{time.strftime('%Y_%m_%d_%H:%M:%S')}_{args.dataset}"

    f_dataset = args.dataset[0].replace("/", "_")
    output_dir = f"{args.output_dir}/{args.train_option}_{f_dataset}_{time.strftime('%m_%d_%H:%M')}"
    # model.config.use_cache = False
    training_args = TrainingArguments(
        output_dir=output_dir,  # 保存模型的路径
        learning_rate=args.learning_rate,  # 学习率
        per_device_train_batch_size=args.batch_size,
        num_train_epochs=args.num_train_epochs,  # 训练 epoch 数
        # weight_decay=0.01,  # 权重衰减
        logging_dir=log_dir,  # 日志路径
        logging_steps=5,
        save_steps=1000,
        save_total_limit=2,  # 最多保存两个模型
        save_strategy="epoch",
        bf16=True,
        lr_scheduler_type="cosine",
        warmup_ratio=0.01,
        dataloader_num_workers=8,
        remove_unused_columns=False,
        # deepspeed=args.ds_config,
    )
    # 只传入 requires_grad 为 True 的参数
    # optimizer = AdamW(
    #     filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate
    # )
    tokenizer.pad_token = tokenizer.eos_token
    # data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator_t,
        # optimizers=(optimizer, None),
    )

    # 开始训练
    trainer.train()
    with open(f"{output_dir}/training_args.json", "w", encoding="utf-8") as f:
        json.dump(training_args.to_dict(), f, indent=4)
    with open(f"{output_dir}/args.json", "w", encoding="utf-8") as json_file:
        args_dict = vars(args)
        json.dump(args_dict, json_file, indent=4)
    if args.train_option == "lora":
        model = model.merge_and_unload()
        model.save_pretrained(f"{output_dir}/lora")
        tokenizer.save_pretrained(f"{output_dir}/lora")

    with open(f"{output_dir}/eval_results.json", "w", encoding="utf-8") as f:
        if trainer.eval_dataset is not None:
            eval_results = trainer.evaluate()
            json.dump(eval_results, f, indent=4)
        else:
            json.dump({"message": "No evaluation dataset provided."}, f, indent=4)
    tokenizer.save_pretrained(output_dir)

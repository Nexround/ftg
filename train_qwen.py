import time
import argparse
import json
from enum import Enum

from functools import partial
from transformers import (
    Trainer,
    TrainingArguments,
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model

from datasets import load_dataset, load_from_disk
import torch
from torch import nn
from src.module.func import (
    unfreeze_ffn_qwen,
    compute_metrics,
    parse_comma_separated,
)


# 定义枚举类型
class TrainOption(Enum):
    LORA = "lora"
    FFN = "ffn"
    FULL = "full"
    C_FULL = "c_full"  # continue full
    TARGET = "target_neurons"
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

    # dataset = load_dataset(*args.dataset, cache_dir="/cache/huggingface/datasets")
    dataset = load_from_disk(*args.dataset)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, cache_dir="/cache/huggingface/hub"
    )
    # tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        cache_dir="/cache/huggingface/hub",
    )
    # tokenizer.chat_template = """{%- if messages[0]['from'] == 'system' %}
    #     {{- '<|im_start|>system\n' + messages[0]['value'] + '<|im_end|>\n' }}
    # {%- else %}
    #     {{- '<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n' }}
    # {%- endif %}
    # {%- for message in messages %}
    #     {%- if message['from'] == 'system' and not loop.first %}
    #         {{- '<|im_start|>system\n' + message['value'] + '<|im_end|>\n' }}
    #     {%- elif message['from'] == 'human' %}
    #         {{- '<|im_start|>user\n' + message['value'] + '<|im_end|>\n' }}
    #     {%- elif message['from'] == 'gpt' and not add_generation_prompt %}
    #         {{- '<|im_start|>assistant\n' + message['value'] + '<|im_end|>\n' }}
    #     {%- endif %}
    # {%- endfor %}
    # {%- if add_generation_prompt %}
    #     {{- '<|im_start|>assistant\n' }}
    # {%- endif %}"""

    def filter_long_examples(example):
        return len(example["input_ids"]) <= 2048  # 仅保留长度 ≤ 512 的样本


    dataset = dataset.filter(filter_long_examples, num_proc=30)
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

    elif args.train_option == "target_neurons":
        print("====================")
        print("=== Training target_neurons ===")
        print("====================")

        for param in model.model.parameters():
            param.requires_grad = False
        for param in model.lm_head.parameters():
            param.requires_grad = False
        with open(args.target_neurons_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        trainable_neurons = list(data)
        print(f"trainable_neurons: {len(trainable_neurons)}")
        hooks = unfreeze_ffn_qwen(model, trainable_neurons)

    elif args.train_option == "ffn":
        print("====================")
        print("=== Training ffn ===")
        print("====================")

        for param in model.bert.parameters():
            param.requires_grad = False
        for param in model.classifier.parameters():
            param.requires_grad = False
        for layer in model.bert.encoder.layer:
            # 获取当前层的中间层和输出层
            intermediate_dense = layer.intermediate.dense
            output_dense = layer.output.dense
            intermediate_dense.weight.requires_grad = True
            intermediate_dense.bias.requires_grad = True
            output_dense.weight.requires_grad = True
            output_dense.bias.requires_grad = True

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
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
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
        fp16=True,
        lr_scheduler_type="cosine",
        warmup_ratio=0.01,
        gradient_accumulation_steps=1,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        # data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    # 开始训练
    trainer.train()
    with open(f"{output_dir}/training_args.json", "w", encoding="utf-8") as f:
        json.dump(training_args.to_dict(), f, indent=4)
    with open(f"{output_dir}/args.json", "w", encoding="utf-8") as json_file:
        args_dict = vars(args)
        json.dump(args_dict, json_file, indent=4)
    if args.train_option == "train_target_neurons":
        for hook in hooks:
            hook.remove()
    if args.train_option == "lora":
        model = model.merge_and_unload()
        model.save_pretrained(f"{output_dir}/lora")
    with open(f"{output_dir}/eval_results.json", "w", encoding="utf-8") as f:
        eval_results = trainer.evaluate()
        json.dump(eval_results, f, indent=4)

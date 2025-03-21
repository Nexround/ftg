import json
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from tqdm.auto import tqdm
from functools import partial
from src.module.loki_linear import replace_all_target_linear_qwen
from torch.amp import autocast

# 参数设置
model_name = "Qwen/Qwen2.5-0.5B-Instruct"  # 预训练模型名称
batch_size = 1  # 批次大小
learning_rate = 1e-5  # 学习率
num_epochs = 1  # 训练轮数

# 检查GPU可用性
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 加载分词器和模型
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    attn_implementation="flash_attention_2",
    cache_dir="/cache/huggingface/hub",
    torch_dtype=torch.bfloat16,
)
model.to(device)
model.train()
for param in model.model.parameters():
    param.requires_grad = False
for param in model.lm_head.parameters():
    param.requires_grad = False
with open("target_neurons/random_neurons.json", "r", encoding="utf-8") as f:
    data = json.load(f)
trainable_neurons = list(data)
# print(f"trainable_neurons: {x:= }")
replace_all_target_linear_qwen(model, trainable_neurons)
dataset = load_dataset("lightblue/reranker_continuous_filt_max7_train")
dataset = dataset.select_columns(["conversations"])
dataset = dataset["train"].shuffle(seed=42)


def data_collator(batch, tokenizer):
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



# 创建DataLoader
train_dataloader = DataLoader(
    dataset, batch_size=batch_size, collate_fn=partial(data_collator, tokenizer = tokenizer)
)

# 优化器设置
optimizer = AdamW(
    filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate
)

# 训练循环
for epoch in range(num_epochs):
    total_loss = 0
    progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}")
    
    for batch in progress_bar:
        batch = {k: v.to(device) for k, v in batch.items()}
        print(f"序列长度：{batch["input_ids"].shape}")
        # 前向传播使用bfloat16
        with autocast(dtype=torch.bfloat16, device_type="cuda"):  # 显式指定bfloat16
            outputs = model(**batch)
            loss = outputs.loss
        
        # 直接反向传播（不使用GradScaler）
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        total_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())

    avg_train_loss = total_loss / len(train_dataloader)
    print(f"Average training loss: {avg_train_loss:.4f}")
# 保存模型
model.save_pretrained("./fine_tuned_model")
tokenizer.save_pretrained("./fine_tuned_model")

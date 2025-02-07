from datasets import load_dataset, load_from_disk, concatenate_datasets
import datasets
from transformers import AutoTokenizer
import torch

model_name = "Qwen/Qwen2.5-0.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)

ds = load_dataset("lightblue/reranker_continuous_filt_max7_train")
# ds = load_from_disk("/root/reranker_continuous_filt_max7_train/data")
ds = ds.select_columns(["conversations"])

print(ds)
# # 定义过滤函数：若文本token数量超过8192则过滤掉该样本
# def filter_by_length(example):
#     conv = example["conversations"]
#     # 转换消息格式（与转换函数中的逻辑一致）
#     messages = []
#     for msg in conv:
#         role = msg["from"]
#         content = msg["value"]
#         if role == "system":
#             messages.append({"role": "system", "content": content})
#         elif role == "human":
#             messages.append({"role": "user", "content": content})
#         elif role == "gpt":
#             messages.append({"role": "assistant", "content": content})
#     # 使用模型自带的模板生成格式化文本
#     text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
#     # 对文本进行编码，注意不进行截断
#     tokenized = tokenizer(text, truncation=False)
#     return len(tokenized["input_ids"]) <= 8192

# # 过滤掉长度超过8192 token 的样本
# ds = ds.filter(filter_by_length, num_proc=20)
# print(f"数据过滤后样本数：{len(ds['train'])}")

def convert_format(batch):
    # 转换消息格式
    batch_converted_messages = []
    for conv in batch["conversations"]:
        converted = []
        for msg in conv:
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
        tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        for messages in batch_converted_messages
    ]
    
    # 批量编码文本
    model_inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        return_tensors="pt",
        return_attention_mask=True
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
            max_length=max_seq_len  # 防止溢出
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


# 应用转换函数
converted_dataset = ds.map(
    convert_format, remove_columns=["conversations"], num_proc=20, batched=True, batch_size=40
)
print(converted_dataset)

converted_dataset.save_to_disk(
    "/cache/huggingface/datasets/reranker_conversations_converted_8192"
)

from datasets import load_dataset, load_from_disk, concatenate_datasets
import datasets
from transformers import AutoTokenizer
import torch

model_name = "Qwen/Qwen2.5-0.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)

ds = load_dataset("lightblue/reranker_continuous_filt_max7_train")
ds = ds.select_columns(["conversations"])

print(ds)


def convert_format(example):
    messages = example["conversations"]
    converted_messages = []

    # 将原始消息转换为目标格式
    for message in messages:
        role = message["from"]
        content = message["value"]
        if role == "system":
            converted_messages.append({"role": "system", "content": content})
        elif role == "human":
            converted_messages.append({"role": "user", "content": content})
        elif role == "gpt":
            converted_messages.append({"role": "assistant", "content": content})

    # 使用 apply_chat_template 格式化对话
    text = tokenizer.apply_chat_template(
        converted_messages, tokenize=False, add_generation_prompt=False
    )

    # 对输入进行编码
    model_inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    input_ids = model_inputs["input_ids"][0]  # 假设 batch_size=1

    # 构造 labels
    # 找到 assistant 消息的起始位置
    assistant_index = len(
        tokenizer.apply_chat_template(
            converted_messages[:-1], tokenize=False, add_generation_prompt=False
        )
    )  # 去掉最后一条 assistant 消息

    # 对 assistant 消息进行编码
    assistant_message = converted_messages[-1]["content"]
    labels_text = (
        assistant_message + tokenizer.eos_token + "\n"
    )  # 使用 EOS token 作为结束符
    labels_ids = tokenizer(labels_text, return_tensors="pt")["input_ids"][0]

    # 初始化 labels 为全 -100（屏蔽）
    labels = torch.full_like(input_ids, -100)
    # print(labels.shape, assistant_index, labels_ids.shape)
    # 将 assistant 部分的 token 设置为 labels_ids
    labels[-len(labels_ids) :] = labels_ids  # assistant_index + len(labels_ids)

    return {
        "input_ids": model_inputs["input_ids"][0],
        "attention_mask": model_inputs["attention_mask"][0],
        "labels": labels,  # 恢复 batch 维度
    }


# 应用转换函数
converted_dataset = ds.map(
    convert_format, remove_columns=["conversations"], num_proc=20
)
print(converted_dataset)

converted_dataset.save_to_disk(
    "/cache/huggingface/datasets/reranker_conversations_converted"
)

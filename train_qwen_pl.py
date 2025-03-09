from pytorch_lightning import LightningDataModule
from datasets import load_dataset
from transformers import AutoTokenizer
import torch
import pytorch_lightning as pl
from transformers import AutoModelForCausalLM
from torch.utils.data import DataLoader
from src.module.loki_linear import replace_all_target_linear_qwen
import json

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


class Qwen2DataModule(LightningDataModule):
    def __init__(self, model_name, batch_size=4):
        super().__init__()
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def prepare_data(self):
        # 加载或预处理数据集（示例使用dummy数据）
        self.dataset = load_dataset("lightblue/reranker_continuous_filt_max7_train")

    def collate_fn(self, batch):
        return data_collator(batch, self.tokenizer)

    def train_dataloader(self):
        return DataLoader(
            self.dataset["train"],
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
            num_workers=39
        )


class Qwen2Train(pl.LightningModule):
    def __init__(self, model_name):
        super().__init__()

        # 加载预训练模型
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        for param in self.model.model.parameters():
            param.requires_grad = False
        for param in self.model.lm_head.parameters():
            param.requires_grad = False
        with open("/workspace/ftg/target_neurons/random_neurons.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        trainable_neurons = list(data)
        # print(f"trainable_neurons: {x:= }")
        replace_all_target_linear_qwen(self.model, trainable_neurons)

    def forward(self, input_ids, attention_mask, labels=None):
        return self.model(input_ids, attention_mask=attention_mask, labels=labels)

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs.loss
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=1e-4)


from pytorch_lightning import Trainer
torch.set_float32_matmul_precision("medium")
datamodule = Qwen2DataModule("Qwen/Qwen2.5-0.5B-Instruct")
model = Qwen2Train("Qwen/Qwen2.5-0.5B-Instruct")

trainer = Trainer(
    max_epochs=1,
    accelerator="auto",
    devices="auto",
    precision="bf16-mixed",  # 混合精度训练节省显存
)

trainer.fit(model, datamodule=datamodule)

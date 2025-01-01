import random
from datasets import load_dataset
from transformers import BertForMaskedLM, BertTokenizer, pipeline
from module.func import extract_random_samples
from tqdm import tqdm
import argparse
from module.func import parse_comma_separated
from typing import List, Tuple, Any
import re

MAX_SEQ_LENGTH = 300
MAX_NUM_TOKENS = MAX_SEQ_LENGTH - 2


def mask_and_truncate_text(
    texts: List[str],
    tokenizer: Any,
    max_length: int = 512,
    mask_token: str = "[MASK]",
    stop_words: List[str] = None
) -> Tuple[List[str], List[int]]:
    """
    对输入文本随机位置添加一个 [MASK] 标签，并根据分词器结果处理截断问题，同时优化 [MASK] 的放置位置。

    Args:
        texts (List[str]): 输入文本列表。
        tokenizer (Any): 分词器对象。
        max_length (int): 截断的最大长度，默认值为 512。
        mask_token (str): 掩码符号，默认值为 "[MASK]"。
        stop_words (List[str]): 语义意义较弱的单词列表，默认值为常见英语停用词。

    Returns:
        Tuple[List[str], List[int]]: 
            - 添加了一个 [MASK] 并截断后的文本列表。
            - 每个文本中被掩盖的单词索引位置列表。
    """
    if stop_words is None:
        # 常见的英语停用词，可根据需要扩展
        stop_words = {"am", "is", "are", "do", "does", "was", "were", "be", "been", "being"}

    masked_texts = []
    masked_positions = []

    for text in texts:
        if not text.strip():  # 跳过空字符串
            continue

        # 对文本进行分词并截断到最大长度
        tokens = tokenizer.tokenize(text)
        truncated_tokens = tokens[:max_length]

        # 确保文本非空
        if not truncated_tokens:
            masked_texts.append("")
            masked_positions.append(-1)
            continue

        # 过滤标点符号和停用词的位置
        valid_positions = [
            i for i, token in enumerate(truncated_tokens)
            if not re.fullmatch(r"\W+", token) and token.lower() not in stop_words
        ]

        if not valid_positions:  # 如果没有有效位置可供放置 [MASK]
            masked_texts.append(tokenizer.convert_tokens_to_string(truncated_tokens))
            masked_positions.append(-1)
            continue

        # 确定 [MASK] 的随机位置
        mask_position = random.choice(valid_positions)

        # 替换指定位置的 token 为 [MASK]
        masked_tokens = [
            mask_token if i == mask_position else token
            for i, token in enumerate(truncated_tokens)
        ]

        # 将 tokens 转换回字符串形式
        masked_sentence = tokenizer.convert_tokens_to_string(masked_tokens)

        masked_texts.append(masked_sentence)
        masked_positions.append(mask_position)

    return masked_texts, masked_positions

# 定义一个函数，用于将文本转换为小写
def lowercase_text(batch):
    batch["text"] = [text.lower() for text in batch["text"]]  # 遍历列表并对每个元素调用 lower()
    return batch

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--max_seq_length",
        default=128,
        type=int,
        help="The maximum total input sequence length after WordPiece tokenization. \n"
        "Sequences longer than this will be truncated, and sequences shorter \n"
        "than this will be padded.",
    )
    parser.add_argument(
        "--batch_size", default=16, type=int, help="Total batch size for cut."
    )
    parser.add_argument(
        "--num_sample", default=10, type=int, help="Num batch of an example."
    )
    parser.add_argument("--dataset", type=parse_comma_separated)

    # parse arguments
    args = parser.parse_args()
    # 1. 加载 Wikitext 数据集
    # dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    dataset = load_dataset(*args.dataset, trust_remote_code=True, cache_dir="/cache/huggingface/datasets")
    dataset = dataset.map(lowercase_text, batched=True)
    dataset = (dataset["train"].shuffle(seed=42).select(range(args.num_sample)))["text"]

    # 2. 加载 BERT tokenizer 和模型
    model_name = "bert-base-uncased"  # 你也可以选择其他的预训练 BERT 模型
    model = BertForMaskedLM.from_pretrained(model_name, cache_dir="/cache/huggingface/transformers")
    tokenizer = BertTokenizer.from_pretrained(model_name)

    # sample_text = extract_random_samples(dataset, args.num_sample)
    masked_texts, masked_positions = mask_and_truncate_text(
        dataset, tokenizer, max_length=MAX_SEQ_LENGTH
    )

    fill_mask = pipeline("fill-mask", model=model, tokenizer=tokenizer)

    batch_size = args.batch_size  # 根据 GPU 内存调整批量大小
    batched_texts = [
        masked_texts[i : i + batch_size] for i in range(0, len(masked_texts), batch_size)
    ]

    predictions = []
    for batch in tqdm(batched_texts, desc="Processing batches"):
        results = fill_mask(batch)
        predictions.extend(results)
    # 7. 计算准确率
    correct_predictions = 0
    total_predictions = 0

    # 计算每个掩蔽位置的准确性
    for idx, masked_sentence in enumerate(masked_texts):
        # 对原始文本进行分词
        original_text = dataset[idx]
        original_tokens = tokenizer.tokenize(original_text)

        # 获取掩蔽位置的原始 token
        mask_position = masked_positions[idx]
        if mask_position == -1 or mask_position >= len(original_tokens):
            # 跳过无效掩码位置
            continue

        masked_token = original_tokens[mask_position]

        # 获取预测的 token
        predicted_token = predictions[idx][0]["token_str"]  # 选择得分最高的预测

        # 如果预测的 token 和原 token 相同，则为正确预测
        if predicted_token == masked_token:
            correct_predictions += 1
        total_predictions += 1
    accuracy = correct_predictions / total_predictions
    print(f"\nAccuracy: {accuracy * 100:.2f}%")

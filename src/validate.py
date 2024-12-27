import random
from datasets import load_dataset
from transformers import BertForMaskedLM, BertTokenizer, pipeline
from module.func import extract_random_samples
from tqdm import tqdm
import argparse
from module.func import parse_comma_separated
from typing import List, Tuple, Any

MAX_SEQ_LENGTH = 300
MAX_NUM_TOKENS = MAX_SEQ_LENGTH - 2


def mask_and_truncate_text(
    texts: List[str],
    tokenizer: Any,
    max_length: int = 512,
    mask_token: str = "[MASK]"
) -> Tuple[List[str], List[int]]:
    """
    对输入文本随机位置添加一个 [MASK] 标签，并根据分词器结果处理截断问题。

    Args:
        texts (List[str]): 输入文本列表。
        tokenizer (Any): 分词器对象。
        max_length (int): 截断的最大长度，默认值为 512。
        mask_token (str): 掩码符号，默认值为 "[MASK]"。

    Returns:
        Tuple[List[str], List[int]]: 
            - 添加了一个 [MASK] 并截断后的文本列表。
            - 每个文本中被掩盖的单词索引位置列表。
    """
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

        # 确定 [MASK] 的随机位置
        mask_position = random.randint(0, len(truncated_tokens) - 1)

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
    dataset = load_dataset(*args.dataset)

    # 2. 加载 BERT tokenizer 和模型
    model_name = "bert-base-uncased"  # 你也可以选择其他的预训练 BERT 模型
    model = BertForMaskedLM.from_pretrained(model_name)
    tokenizer = BertTokenizer.from_pretrained(model_name)

    sample_text = extract_random_samples(dataset, args.num_sample)
    masked_texts, masked_positions = mask_and_truncate_text(
        sample_text, tokenizer, max_length=MAX_SEQ_LENGTH
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
        original_text = sample_text[idx]
        words = original_text.split()

        # 获取掩蔽位置的原始词
        mask_position = masked_positions[idx]
        masked_word = words[mask_position]

        # 获取预测的单词
        predicted_word = predictions[idx][0]["token_str"]  # 选择得分最高的预测

        # 如果预测的单词和原单词相同，则为正确预测
        if predicted_word == masked_word:
            correct_predictions += 1
        total_predictions += 1

    accuracy = correct_predictions / total_predictions
    print(f"\nAccuracy: {accuracy * 100:.2f}%")

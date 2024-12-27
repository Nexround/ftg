import random
from datasets import load_dataset
from transformers import BertForMaskedLM, BertTokenizer, pipeline
from module.func import extract_random_samples
from tqdm import tqdm
import argparse
from module.func import parse_comma_separated

MAX_SEQ_LENGTH = 300
MAX_NUM_TOKENS = MAX_SEQ_LENGTH - 2


def mask_and_truncate_text(texts, tokenizer, max_length: int = 512) -> str:
    """
    对输入文本随机位置添加一个 [MASK] 标签，并根据分词器结果处理截断问题。

    Args:
        text (str): 输入文本。
        tokenizer_name (str): 分词器的名称，默认值为 "bert-base-uncased"。
        max_length (int): 截断的最大长度，默认值为 512。

    Returns:
        str: 添加了一个 [MASK] 并截断后的文本。
    """
    masked_texts = []
    masked_positions = []  # 保存每个句子中被掩盖的位置，便于后续准确率计算
    # 将文本分词为 token
    for text in texts:
        # tokens = tokenizer.tokenize(text)
        truncated_text = text.split()[:MAX_NUM_TOKENS]

        # 截断到最大长度
        # truncated_tokens = tokens[:max_length]
        # truncated_text = tokenizer.convert_tokens_to_string(truncated_tokens)
        # truncated_words = truncated_text.split()[:MAX_NUM_TOKENS]
        # 确保只添加一个 [MASK]
        mask_position = random.randint(0, len(truncated_text) - 1)
        masked_sentence = " ".join(
            [
                word if i != mask_position else "[MASK]"
                for i, word in enumerate(truncated_text)
            ]
        )
        masked_positions.append(mask_position)
        masked_texts.append(masked_sentence)

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

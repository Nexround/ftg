import random
from datasets import load_dataset
from transformers import BertForMaskedLM, BertTokenizer, pipeline
from module.func import extract_random_samples

# 1. 加载 Wikitext 数据集
# dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
dataset = load_dataset("imdb")

# 2. 加载 BERT tokenizer 和模型
model_name = "bert-base-uncased"  # 你也可以选择其他的预训练 BERT 模型
model = BertForMaskedLM.from_pretrained(model_name)
tokenizer = BertTokenizer.from_pretrained(model_name)

sample_text = extract_random_samples(dataset, 100)

# 4. 为了测试准确率，随机将句子中的一个词替换为 [MASK]
masked_texts = []
masked_positions = []  # 保存每个句子中被掩盖的位置，便于后续准确率计算

for text in sample_text:
    words = text.split()
    if len(words) < 2:  # 确保句子至少有两个词
        continue

    # 随机选择一个词的位置进行掩盖
    mask_position = random.randint(0, len(words) - 1)
    masked_positions.append(mask_position)

    # 替换选中的位置为 [MASK]
    masked_sentence = " ".join([
        word if i != mask_position else "[MASK]" for i, word in enumerate(words)
    ])
    masked_texts.append(masked_sentence)

# 5. 使用填充任务的 pipeline 进行推理
fill_mask = pipeline("fill-mask", model=model, tokenizer=tokenizer, truncation=True)

# 6. 对每个掩蔽句子进行推理并保存结果
predictions = []
for masked_sentence in masked_texts:
    result = fill_mask(masked_sentence)
    predictions.append(result)

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

from datasets import load_dataset
from transformers import BertForMaskedLM, BertTokenizer, pipeline
from modules.func import extract_random_samples

# 1. 加载 Wikitext 数据集
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

# 2. 加载 BERT tokenizer 和模型
model_name = "bert-base-uncased"  # 你也可以选择其他的预训练 BERT 模型
model = BertForMaskedLM.from_pretrained(model_name)
tokenizer = BertTokenizer.from_pretrained(model_name)

sample_text = extract_random_samples(dataset, 100)

# 4. 为了测试准确率，手动将句子中的一个词替换为 [MASK]
masked_texts = []
for text in sample_text:

    words = text.split()
    # 随机选择一个词替换为 [MASK]，这里只替换第一个词
    masked_sentence = " ".join([word if i != 0 else "[MASK]" for i, word in enumerate(words)])  # 第一个词替换为 [MASK]
    masked_texts.append(masked_sentence)

# 打印掩蔽后的句子（便于调试）
# print("Masked sentences:")
# for mt in masked_texts:
#     print(mt,"\n")

# 5. 使用填充任务的 pipeline 进行推理
fill_mask = pipeline("fill-mask", model=model, tokenizer=tokenizer)

# 6. 对每个掩蔽句子进行推理并保存结果
predictions = []
for masked_sentence in masked_texts:
    result = fill_mask(masked_sentence)
    predictions.append(result)

# 打印预测结果
# print("\nPredictions:")
# for idx, prediction in enumerate(predictions):
#     print(f"\nMasked sentence {idx + 1}: {masked_texts[idx]}")
#     for pred in prediction:
#         print(f"Predicted word: {pred['token_str']} - with score: {pred['score']}")

# 7. 计算准确率
correct_predictions = 0
total_predictions = 0

# 计算每个掩蔽位置的准确性
for idx, masked_sentence in enumerate(masked_texts):
    original_text = sample_text[idx]
    # 获取掩蔽位置的原始词（假设第一个词被替换为 [MASK]）
    words = original_text.split()
    masked_word = words[0]  # 这里只考虑第一个被掩蔽的词
    
    # 获取预测的单词
    predicted_word = predictions[idx][0]["token_str"]  # 选择得分最高的预测
    
    # 如果预测的单词和原单词相同，则为正确预测
    if predicted_word == masked_word:
        correct_predictions += 1
    total_predictions += 1

accuracy = correct_predictions / total_predictions
print(f"\nAccuracy: {accuracy * 100:.2f}%")

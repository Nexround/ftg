from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import torch
from tqdm import tqdm
import numpy as np
import json
# 配置参数
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"  # 例如 "meta-llama/Llama-2-7b-hf"
mmlu_all_sets = [
    'college_biology',
    'college_chemistry',
    'college_computer_science',
    'college_mathematics',
    'college_physics',
    'electrical_engineering',
    'astronomy',
    'anatomy',
    'abstract_algebra',
    'machine_learning',
    'clinical_knowledge',
    'global_facts',
    'management',
    'nutrition',
    'marketing',
    'professional_accounting',
    'high_school_geography',
    'international_law',
    'moral_scenarios',
    'computer_security',
    'high_school_microeconomics',
    'professional_law',
    'medical_genetics',
    'professional_psychology',
    'jurisprudence',
    'world_religions',
    'philosophy',
    'virology',
    'high_school_chemistry',
    'public_relations',
    'high_school_macroeconomics',
    'human_sexuality',
    'elementary_mathematics',
    'high_school_physics',
    'high_school_computer_science',
    'high_school_european_history',
    'business_ethics',
    'moral_disputes',
    'high_school_statistics',
    'miscellaneous',
    'formal_logic',
    'high_school_government_and_politics',
    'prehistory',
    'security_studies',
    'high_school_biology',
    'logical_fallacies',
    'high_school_world_history',
    'professional_medicine',
    'high_school_mathematics',
    'college_medicine',
    'high_school_us_history',
    'sociology',
    'econometrics',
    'high_school_psychology',
    'human_aging',
    'us_foreign_policy',
    'conceptual_physics',
]
NUM_SHOTS = 5  # 5-shot学习
MAX_LENGTH = 2048  # 模型最大上下文长度
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 加载模型和tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
)


def build_conversation(subset, train_samples, test_sample):
    conversation = []
    subject = subset.replace("_", " ").title()

    # 添加few-shot示例
    for example in train_samples:
        # 用户问题
        human_msg = {
            "role": "HUMAN",
            "prompt": f"There is a single choice question about {subject}. Answer the question by replying A, B, C or D.\n"
            f"Question: {example['question']}\n"
            f"Choices:\n"
            + "\n".join(
                [
                    f"{chr(65+i)}. {choice}"
                    for i, choice in enumerate(example["choices"])
                ]
            )
            + "\nAnswer: \n",
        }

        # 模型回答
        bot_msg = {"role": "BOT", "prompt": f"{chr(65+example['answer'])}\n"}

        conversation.extend([human_msg, bot_msg])

    # 添加测试问题
    test_human_msg = {
        "role": "HUMAN",
        "prompt": f"There is a single choice question about {subject}. Answer the question by replying A, B, C or D.\n"
        f"Question: {test_sample['question']}\n"
        f"Choices:\n"
        + "\n".join(
            [
                f"{chr(65+i)}. {choice}"
                for i, choice in enumerate(test_sample["choices"])
            ]
        )
        + "\nAnswer: \n",
    }
    conversation.append(test_human_msg)

    return conversation


def generate_response(conversation):
    # 将对话历史转换为模型输入格式
    inputs = []
    for msg in conversation:
        if msg["role"] == "HUMAN":
            inputs.append({"role": "user", "content": msg["prompt"]})
        elif msg["role"] == "BOT":
            inputs.append({"role": "assistant", "content": msg["prompt"]})

    # 使用模型的聊天模板
    input_ids = tokenizer.apply_chat_template(
        inputs,
        add_generation_prompt=True,
        return_tensors="pt",
        # tokenize=False
    ).to(DEVICE)
    # model_input_ids = tokenizer([input_ids], return_tensors="pt").to(DEVICE)
    # 生成回答
    generated_ids = model.generate(
        input_ids,
        max_new_tokens=10,
    )
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(input_ids, generated_ids)]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    # 解码并提取答案
    return response.strip()  # 提取第一个字母


def evaluate_subset(subset):
    dataset = load_dataset("cais/mmlu", subset)
    test_data = dataset["test"]
    few_shot_samples = dataset["dev"]

    correct = 0
    total = len(test_data)

    for test_sample in tqdm(test_data, desc=f"Evaluating {subset}"):

        # 构建对话历史
        conversation = build_conversation(subset, few_shot_samples, test_sample)

        # 生成模型响应
        prediction = generate_response(conversation)
        
        # 检查prediction是否在ABCD中
        # if prediction not in {"A", "B", "C", "D"}:
        #     raise ValueError(f"Invalid prediction: {prediction}")

        # 验证答案
        if prediction == (answer:=f"{chr(65+test_sample["answer"])}"):
            correct += 1

    accuracy = correct / total
    return accuracy


if __name__ == "__main__":
    results = {}
    for subset in mmlu_all_sets:
        accuracy = evaluate_subset(subset)
        results[subset] = accuracy
        print(f"{subset} Accuracy: {accuracy:.2%}")

    # 保存详细结果
    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)

import re
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import json
import torch
import os


def get_last_folder(path):
    normalized_path = os.path.normpath(path)
    return os.path.basename(normalized_path)


# 配置参数
MODEL_NAMES = [
    "/cache/models/suppressed_5_L",
    "/cache/models/suppressed_5",
    "/cache/models/suppressed_10_L",
    # "/cache/models/suppressed_10",
]
mmlu_all_sets = [
    "college_biology",
    "college_chemistry",
    "college_computer_science",
    "college_mathematics",
    "college_physics",
    "electrical_engineering",
    "astronomy",
    "anatomy",
    "abstract_algebra",
    "machine_learning",
    "clinical_knowledge",
    "global_facts",
    "management",
    "nutrition",
    "marketing",
    "professional_accounting",
    "high_school_geography",
    "international_law",
    "moral_scenarios",
    "computer_security",
    "high_school_microeconomics",
    "professional_law",
    "medical_genetics",
    "professional_psychology",
    "jurisprudence",
    "world_religions",
    "philosophy",
    "virology",
    "high_school_chemistry",
    "public_relations",
    "high_school_macroeconomics",
    "human_sexuality",
    "elementary_mathematics",
    "high_school_physics",
    "high_school_computer_science",
    "high_school_european_history",
    "business_ethics",
    "moral_disputes",
    "high_school_statistics",
    "miscellaneous",
    "formal_logic",
    "high_school_government_and_politics",
    "prehistory",
    "security_studies",
    "high_school_biology",
    "logical_fallacies",
    "high_school_world_history",
    "professional_medicine",
    "high_school_mathematics",
    "college_medicine",
    "high_school_us_history",
    "sociology",
    "econometrics",
    "high_school_psychology",
    "human_aging",
    "us_foreign_policy",
    "conceptual_physics",
]
NUM_SHOTS = 5
MAX_LENGTH = 4196


def build_prompt(subset, train_samples, test_sample, tokenizer):
    messages = []
    subject = subset.replace("_", " ").title()

    for example in train_samples:
        human_content = (
            f"There is a single choice question about {subject}. Answer the question by replying A, B, C or D.\n"
            f"Question: {example['question']}\n"
            "Choices:\n"
            + "\n".join(
                [
                    f"{chr(65 + i)}. {choice}"
                    for i, choice in enumerate(example["choices"])
                ]
            )
            + "\nAnswer: \n"
        )
        messages.append({"role": "user", "content": human_content})

        bot_content = f"{chr(65 + example['answer'])}\n"
        messages.append({"role": "assistant", "content": bot_content})

    test_content = (
        f"There is a single choice question about {subject}. Answer the question by replying A, B, C or D.\n"
        f"Question: {test_sample['question']}\n"
        "Choices:\n"
        + "\n".join(
            [
                f"{chr(65 + i)}. {choice}"
                for i, choice in enumerate(test_sample["choices"])
            ]
        )
        + "\nAnswer: \n"
    )
    messages.append({"role": "user", "content": test_content})

    prompt = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    return prompt


def batch_generate(prompts, llm):
    sampling_params = SamplingParams(max_tokens=2, temperature=0.0, stop=["\n"])
    outputs = llm.generate(prompts, sampling_params)

    predictions = []
    for output in outputs:
        response = output.outputs[0].text.strip()
        print(f"Response: {response}")
        predictions.append(response)
        # for char in response:
        #     if char in {"A", "B", "C", "D"}:
        #         predictions.append(char)
        #         break
        # else:
        #     predictions.append("")
    return predictions



def evaluate_subset(subset, tokenizer, llm):
    dataset = load_dataset("cais/mmlu", subset)
    test_data = dataset["test"]
    few_shot_samples = dataset["dev"].shuffle().select(range(NUM_SHOTS))

    prompts = [
        build_prompt(subset, few_shot_samples, ts, tokenizer) for ts in test_data
    ]
    predictions = batch_generate(prompts, llm)

    correct = 0
    for pred, test_sample in zip(predictions, test_data):
        target = chr(65 + test_sample["answer"])
        model_answer = pred.upper()[0]
        if model_answer == target:
            correct += 1

    return correct / len(test_data)

if __name__ == "__main__":
    for model_name in MODEL_NAMES:
        print(f"\n{'=' * 40}")
        print(f"Evaluating model: {model_name}")
        print(f"{'=' * 40}\n")

        last_folder = get_last_folder(model_name)
        results = {}

        # 初始化当前模型的组件
        llm = LLM(
            model=model_name,
            tensor_parallel_size=torch.cuda.device_count(),
            dtype="bfloat16" if torch.cuda.is_available() else "auto",
            max_model_len=MAX_LENGTH,
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # 执行评估
        for subset in mmlu_all_sets:
            try:
                accuracy = evaluate_subset(subset, tokenizer, llm)
                results[subset] = accuracy
                print(f"{subset.ljust(35)} Accuracy: {accuracy:.2%}")
            except Exception as e:
                print(f"Error evaluating {subset}: {str(e)}")
                results[subset] = None

        # 计算平均精度
        valid_results = [v for v in results.values() if v is not None]
        mmlu_average = sum(valid_results) / len(valid_results) if valid_results else 0
        results["mmlu_average"] = mmlu_average
        print(f"\nAverage MMLU Accuracy: {mmlu_average:.2%}")

        # 保存结果
        output_file = f"{last_folder}_mmlu_results.json"
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {output_file}")

        # 清理资源
        del llm, tokenizer
        torch.cuda.empty_cache()

import random
import time
import argparse
import logging
import os

import jsonlines
import torch
import numpy as np
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer
from src.module.func import (
    convert_to_triplet_ig_top,
    parse_comma_separated,
)
from src.module.Qwen2Model import CustomQwen2ForCausalLM
from pprint import pprint

# set logger
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_path",
        default=None,
        type=str,
        required=True,
        help="Bert pre-trained model selected in the list: bert-base-uncased, "
        "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    # Other parameters
    parser.add_argument(
        "--max_seq_length",
        default=128,
        type=int,
        help="The maximum total input sequence length after WordPiece tokenization. \n"
        "Sequences longer than this will be truncated, and sequences shorter \n"
        "than this will be padded.",
    )

    parser.add_argument("--gpus", type=str, default="0", help="available gpus id")
    parser.add_argument(
        "--seed", type=int, default=42, help="random seed for initialization"
    )

    parser.add_argument(
        "--times", default=10, type=int, help="Total batch size for cut."
    )
    parser.add_argument("--num_sample", default=10000, type=int)

    parser.add_argument("--retention_threshold", default=99, type=int)
    parser.add_argument("--result_file", type=str)
    parser.add_argument("--dataset", type=parse_comma_separated)

    # parse arguments
    args = parser.parse_args()

    def get_gradient_size(model):
        grad_size = sum(
            p.grad.element_size() * p.grad.numel()
            for p in model.parameters()
            if p.grad is not None
        )
        return grad_size / 1024**2  # 转换为 MB

    device = torch.device("cuda:0")
    n_gpu = 1

    print(
        "device: {} n_gpu: {}, distributed training: {}".format(
            device, n_gpu, bool(n_gpu > 1)
        )
    )

    # set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    # save args
    os.makedirs(args.output_dir, exist_ok=True)
    print("***** CUDA.empty_cache() *****")
    torch.cuda.empty_cache()
    # quantization_config = BitsAndBytesConfig(load_in_4bit=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = CustomQwen2ForCausalLM.from_pretrained(
        args.model_path,
        # quantization_config=quantization_config,
        # torch_dtype="auto",
        torch_dtype=torch.bfloat16,
        # attn_implementation="flash_attention_2",
        device_map="auto",
    )
    # model.gradient_checkpointing_enable() # 目前看来没什么用
    # model.model.embed_tokens.to("cpu")
    # model.lm_head.to("cpu")
    # model = torch.compile(model)
    # print(model.get_memory_footprint())

    # data parallel
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # model.eval()
    def get_model_size(model):
        param_size = sum(p.element_size() * p.numel() for p in model.parameters())
        return param_size / 1024**2  # 转换为 MB

    print(f"Model Weights Memory: {get_model_size(model):.2f} MB")

    def build_conversation(subset, train_samples, test_sample):
        conversation = []
        subject = subset.replace("_", " ").title()

        # 添加few-shot示例
        for example in train_samples:
            # 用户问题
            human_msg = {
                "role": "user",
                "content": f"There is a single choice question about {subject}. Answer the question by replying A, B, C or D.\n"
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
            bot_msg = {"role": "assistant", "content": f"{chr(65+example['answer'])}\n"}

            conversation.extend([human_msg, bot_msg])

        # 添加测试问题
        test_human_msg = {
            "role": "user",
            "content": f"There is a single choice question about {subject}. Answer the question by replying A, B, C or D.\n"
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


    record_list = []
    for subset in mmlu_all_sets:
        dataset = load_dataset("cais/mmlu", subset)
        test_dataset = dataset["test"]
        few_shot_samples = dataset["dev"]
        for test_sample in tqdm(test_dataset, desc=f"Evaluating {subset}"):

            # 构建对话历史
            conversation = build_conversation(subset, few_shot_samples, test_sample)
            inputs = tokenizer.apply_chat_template(
                conversation,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True, # 返回input_ids和attention_mask
            ).to(device)

            # record running time
            tic = time.perf_counter()

            ig_dict = {"ig_gold": []}

            logits = model.forward(
                **(inputs),
                target_token_idx=-1,
                use_cache=True,
            )
            # logits = outputs.logits
            predicted_class = int(torch.argmax(logits, dim=-1))  # 预测类别

            model.forward_with_partitioning(target_token_idx=-1, times=args.times)
            ig_gold = model.calculate_integrated_gradients(target_label=predicted_class)

            for ig in ig_gold:
                # 为batch inference预留的for
                ig_dict["ig_gold"].append(ig)

            ig_dict["ig_gold"] = convert_to_triplet_ig_top(
                ig_dict["ig_gold"], args.retention_threshold
            )
            record_list.append([ig_dict])
            # record running time
            toc = time.perf_counter()
            print(f"***** Costing time: {toc - tic:0.4f} seconds *****")
            # pprint(torch.cuda.memory_stats()) #没什么用
            # print(f"Gradients Memory: {get_gradient_size(model):.2f} MB")

            model.clean()

    with jsonlines.open(os.path.join(args.output_dir, args.result_file), "w") as fw:
        fw.write(record_list)

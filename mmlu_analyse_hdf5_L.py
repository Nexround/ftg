import random
import time
import argparse
import os

import torch
import numpy as np
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer
from src.module.func import (
    save_array_to_hdf5,
    parse_comma_separated,
)
from src.module.Llama3Model import CustomLlamaForCausalLM

torch.set_float32_matmul_precision("medium")

mmlu_all_sets = [
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
# for subset in tqdm(mmlu_all_sets):
#     load_dataset("cais/mmlu", subset)

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

    parser.add_argument("--result_file", type=str)
    parser.add_argument("--write_mode", type=str)
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
    model = CustomLlamaForCausalLM.from_pretrained(
        args.model_path,
        # quantization_config=quantization_config,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = torch.compile(model)

    # model.gradient_checkpointing_enable() # 目前看来没什么用
    # model.model.embed_tokens.to("cpu")
    # model.lm_head.to("cpu")
    print(model.get_memory_footprint())

    # data parallel
    # if n_gpu > 1:
    #     model = torch.nn.DataParallel(model)

    # model.eval()
    def get_model_size(model):
        param_size = sum(p.element_size() * p.numel() for p in model.parameters())
        return param_size / 1024**2  # 转换为 MB

    print(f"Model Weights Memory: {get_model_size(model):.2f} MB")
    def build_conversation(subset, test_sample):
        conversation = []
        subject = subset.replace("_", " ").title()

        # 直接添加测试问题
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


    # fw = jsonlines.open(
    #     os.path.join(args.output_dir, args.result_file),
    #     mode=args.write_mode if args.write_mode is not None else "w",
    # )
    for subset in tqdm(mmlu_all_sets, desc="📦"):
        dataset = load_dataset("cais/mmlu", subset)
        test_dataset = dataset["test"]
        for idx, test_sample in tqdm(
            enumerate(test_dataset),
            desc=f"🗂️ Evaluating {subset}",
            total=len(test_dataset),
            leave=False,
        ):

            # 构建对话历史
            conversation = build_conversation(subset, test_sample)
            inputs = tokenizer.apply_chat_template(
                conversation,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,  # 返回input_ids和attention_mask
            ).to(device)
            inputs["attention_mask"] = inputs["attention_mask"].to(torch.int8)
            # record running time
            tic = time.perf_counter()

            # ig_dict = {"dataset_subset": subset, "idx": idx, "mvp": []}
            logits = model.forward(
                **(inputs),
                target_token_idx=-1,
                # use_cache=False,
            )
            predicted_label = int(torch.argmax(logits, dim=-1))  # 预测类别
            print(tokenizer.decode([predicted_label]))
            model.forward_with_partitioning(
                target_token_idx=-1, times=args.times, predicted_label=predicted_label
            )
            mvp = model.integrated_gradients
            save_array_to_hdf5(os.path.join(args.output_dir, args.result_file), mvp)
            toc = time.perf_counter()
            print(f"***** Costing time: {toc - tic:0.4f} seconds *****")
            model.clean()

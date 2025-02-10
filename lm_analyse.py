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
from transformers import AutoTokenizer, BitsAndBytesConfig
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
    # init tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    # , do_lower_case=args.do_lower_case
    # Load pre-trained BERT
    print("***** CUDA.empty_cache() *****")
    torch.cuda.empty_cache()
    # quantization_config = BitsAndBytesConfig(load_in_4bit=True)

    model = CustomQwen2ForCausalLM.from_pretrained(
        args.model_path,
        # quantization_config=quantization_config,
        torch_dtype="auto",
        attn_implementation="flash_attention_2",
        device_map="auto",
    )
    # model.gradient_checkpointing_enable() # 目前看来没什么用
    # model.model.embed_tokens.to("cpu")
    # model.lm_head.to("cpu")
    # model = torch.compile(model)
    print(model.get_memory_footprint())

    # data parallel
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # model.eval()
    def get_model_size(model):
        param_size = sum(p.element_size() * p.numel() for p in model.parameters())
        return param_size / 1024**2  # 转换为 MB

    print(f"Model Weights Memory: {get_model_size(model):.2f} MB")

    dataset = load_dataset(
        *args.dataset, trust_remote_code=True, cache_dir="/cache/huggingface/datasets"
    )

    # def tokenize_function(examples):
    #     # print(examples["instruction"])
    #     messages = [
    #         {
    #             "role": "system",
    #             "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
    #         },
    #         {"role": "user", "content": examples["instruction"]},
    #     ]
    #     text = tokenizer.apply_chat_template(
    #         messages, tokenize=False, add_generation_prompt=True
    #     )
    #     return tokenizer(text, return_tensors="pt")

    dataset = dataset["train"].shuffle(seed=42).select(range(args.num_sample))
    # tokenized_train = dataset.map(tokenize_function, batched=True, num_proc=32)
    # evaluate args.debug bags for each relation

    record_list = []
    for item in tqdm(dataset):
        # record running time
        prompt = item["instruction"]
        tic = time.perf_counter()
        messages = [
            {
                "role": "system",
                "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
            },
            {"role": "user", "content": prompt},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        model_inputs = tokenizer([text], return_tensors="pt")
        # model_input = model_input.to(model.device)
        ig_dict = {"ig_gold": []}
        end_pos = len(model_inputs.data["input_ids"][0]) - 1
        if end_pos > 100:
            continue
        model_inputs.to(model.device)
        print(end_pos)
        # original pred prob
        logits = model.forward(
            **(model_inputs.data),
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
        print(f"Gradients Memory: {get_gradient_size(model):.2f} MB")

        model.clean()

    with jsonlines.open(os.path.join(args.output_dir, args.result_file), "w") as fw:
        fw.write(record_list)

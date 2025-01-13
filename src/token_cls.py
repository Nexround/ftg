"""
BERT MLM runner
"""

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
from transformers import BertTokenizer
from module.func import (
    convert_to_triplet_ig_top,
    parse_comma_separated,
)
from model import CustomBertForSequenceClassification
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
        "--bert_model",
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
        "--batch_size", default=16, type=int, help="Total batch size for cut."
    )
    parser.add_argument("--num_sample", default=10000, type=int)

    parser.add_argument("--retention_threshold", default=99, type=int)
    parser.add_argument("--result_file", type=str)
    parser.add_argument("--dataset", type=parse_comma_separated)

    # parse arguments
    args = parser.parse_args()

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
    tokenizer = BertTokenizer.from_pretrained(args.bert_model)
    # , do_lower_case=args.do_lower_case
    # Load pre-trained BERT
    print("***** CUDA.empty_cache() *****")
    torch.cuda.empty_cache()
    model = CustomBertForSequenceClassification.from_pretrained(
        args.bert_model, cache_dir="/cache/huggingface/hub"
    ).half()
    model.to(device)

    # data parallel
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)
    model.eval()

    dataset = load_dataset(
        *args.dataset, trust_remote_code=True, cache_dir="/cache/huggingface/datasets"
    )

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=args.max_seq_length,
        )

    dataset = dataset["train"].shuffle(seed=42).select(range(args.num_sample))
    tokenized_train = dataset.map(tokenize_function, batched=True, num_proc=32)
    # evaluate args.debug bags for each relation

    record_list = []

    for item in tqdm(tokenized_train):
        # record running time
        tic = time.perf_counter()
        gold_class = item["label"]
        cls_id = tokenizer.convert_tokens_to_ids("[CLS]")
        # record various results
        ig_dict = {"ig_gold": []}

        # Move input tensors to the same device as the model
        input_ids = torch.tensor(item["input_ids"]).unsqueeze(0).to(device)
        attention_mask = torch.tensor(item["attention_mask"]).unsqueeze(0).to(device)

        cls_pos = 0
        # cls_pos = input_ids.index(cls_id)

        # original pred prob
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            target_token_idx=cls_pos,
        )
        logits = outputs.logits
        predicted_class = int(torch.argmax(logits, dim=-1))  # 预测类别
        if gold_class != predicted_class:
            model.clean()
            continue
        """
        若预测类别与真实类别不一致，则跳过
        只处理[cls]位置上的ig
        
        """
        # predicted_class = torch.argmax(logits, dim=-1).item()
        # print(f"Predicted class: {predicted_class}")
        # label_map = model.config.id2label
        # print(f"Predicted label: {label_map[predicted_class]}")
        model.forward_with_partitioning(target_position=cls_pos)
        ig_gold = model.calulate_integrated_gradients(target_label=gold_class)
        for ig in ig_gold:
            # 为batch inference预留的for
            ig = ig.cpu().detach()
            ig_dict["ig_gold"].append(ig)

        ig_dict["ig_gold"] = convert_to_triplet_ig_top(
            ig_dict["ig_gold"], args.retention_threshold
        )
        record_list.append([ig_dict])
        # record running time
        toc = time.perf_counter()
        print(f"***** Costing time: {toc - tic:0.4f} seconds *****")
        # pprint(torch.cuda.memory_stats())
        model.clean()

    with jsonlines.open(os.path.join(args.output_dir, args.result_file), "w") as fw:
        fw.write(record_list)

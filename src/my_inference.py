"""
BERT MLM runner
"""

import ast
import random
import json
import time
import argparse
import logging
import os
import copy

import jsonlines
import torch
import torch.nn.functional as F
import numpy as np
from datasets import load_dataset
from tqdm import tqdm
from transformers import BertTokenizer
from custom_bert import BertForMaskedLM
from module.func import (
    scaled_input,
    convert_to_triplet_ig_top,
    extract_random_samples,
    parse_comma_separated,
)
from model import CustomBertForMaskedLM
from pprint import pprint
# set logger
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def example2feature(example, max_seq_length, tokenizer):
    """将示例转换为输入特征"""
    features = []
    tokenslist = []

    ori_tokens = tokenizer.tokenize(example)  # 对文本进行分词
    # 如果文本长度超过最大序列长度，进行截断
    if len(ori_tokens) > max_seq_length - 2:
        ori_tokens = ori_tokens[: max_seq_length - 2]

    # 在文本中随机选择一个位置添加 [MASK]
    mask_position = random.randint(
        1, len(ori_tokens) - 1
    )  # 避免将 [MASK] 添加到 [CLS] 或 [SEP] 位置
    gold_obj = ori_tokens[mask_position]  # 原始 token 作为 gold_obj
    ori_tokens[mask_position] = "[MASK]"  # 在随机位置插入 [MASK]

    # 添加特殊字符 ([CLS], [SEP])
    tokens = ["[CLS]"] + ori_tokens + ["[SEP]"]
    base_tokens = ["[UNK]"] + ["[UNK]"] * len(ori_tokens) + ["[UNK]"]
    segment_ids = [0] * len(tokens)

    # 生成 token ID 和 attention mask
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    baseline_ids = tokenizer.convert_tokens_to_ids(base_tokens)
    input_mask = [1] * len(input_ids)

    # 填充 [PAD] 到最大序列长度
    padding = [0] * (max_seq_length - len(input_ids))
    input_ids += padding
    baseline_ids += padding
    segment_ids += padding
    input_mask += padding

    # 断言填充后的长度是正确的
    assert len(baseline_ids) == max_seq_length
    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    features = {
        "input_ids": input_ids,
        "input_mask": input_mask,
        "segment_ids": segment_ids,
        "baseline_ids": baseline_ids,
    }
    tokens_info = {
        "tokens": tokens,
        "gold_obj": gold_obj,  # 保存掩码位置的原始 token 作为 gold_obj
        "pred_obj": None,  # 预测的对象，这里是占位符
    }
    return features, tokens_info


RETENTION_THRESHOLD = 99


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
    parser.add_argument(
        "--do_lower_case",
        default=False,
        action="store_true",
        help="Set this flag if you are using an uncased model",
    )
    parser.add_argument(
        "--no_cuda",
        default=False,
        action="store_true",
        help="Whether not to use CUDA when available",
    )
    parser.add_argument("--gpus", type=str, default="0", help="available gpus id")
    parser.add_argument(
        "--seed", type=int, default=42, help="random seed for initialization"
    )
    parser.add_argument(
        "--debug",
        type=int,
        default=-1,
        help="How many examples to debug. -1 denotes no debugging",
    )

    # parameters about integrated grad
    parser.add_argument(
        "--get_pred", action="store_true", help="Whether to get prediction results."
    )
    parser.add_argument(
        "--get_ig_pred",
        action="store_true",
        help="Whether to get integrated gradient at the predicted label.",
    )
    parser.add_argument(
        "--get_ig_gold",
        action="store_true",
        help="Whether to get integrated gradient at the gold label.",
    )
    parser.add_argument(
        "--get_base", action="store_true", help="Whether to get base values. "
    )
    parser.add_argument(
        "--batch_size", default=16, type=int, help="Total batch size for cut."
    )
    parser.add_argument(
        "--num_batch", default=10, type=int, help="Num batch of an example."
    )
    parser.add_argument(
        "--num_sample", default=10, type=int, help="Num batch of an example."
    )
    parser.add_argument(
        "--retention_threshold", default=99, type=int, help="Num batch of an example."
    )
    parser.add_argument("--result_file", type=str)
    parser.add_argument("--dataset", type=parse_comma_separated)

    # parse arguments
    args = parser.parse_args()

    # set device
    if args.no_cuda or not torch.cuda.is_available():
        device = torch.device("cpu")
        n_gpu = 0
    else:
        device = torch.device("cuda:%s" % args.gpus)
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
    tokenizer = BertTokenizer.from_pretrained(
        args.bert_model, do_lower_case=args.do_lower_case
    )

    # Load pre-trained BERT
    print("***** CUDA.empty_cache() *****")
    torch.cuda.empty_cache()
    model = CustomBertForMaskedLM.from_pretrained(
        args.bert_model, cache_dir="/cache/huggingface/hub"
    )
    model.to(device)

    # data parallel
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)
    model.eval()

    dataset = load_dataset(
        *args.dataset, trust_remote_code=True, cache_dir="/cache/huggingface/datasets"
    )
    # dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    # sample_text = [i for i in dataset["train"]["text"][10:15] if i.strip() != ""]
    # sample_text = extract_random_samples(dataset, args.num_sample)
    dataset = (dataset["train"].shuffle(seed=42).select(range(args.num_sample)))["text"]
    # evaluate args.debug bags for each relation

    res_dict_bag = []
    for text in tqdm(dataset):
        # record running time
        tic = time.perf_counter()
        try:
            eval_features, tokens_info = example2feature(
                text, args.max_seq_length, tokenizer
            )
        except:
            continue
        # convert features to long type tensors
        baseline_ids, input_ids, input_mask, segment_ids = (
            eval_features["baseline_ids"],
            eval_features["input_ids"],
            eval_features["input_mask"],
            eval_features["segment_ids"],
        )
        baseline_ids = torch.tensor(baseline_ids, dtype=torch.long).unsqueeze(0)
        input_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)
        input_mask = torch.tensor(input_mask, dtype=torch.long).unsqueeze(0)
        segment_ids = torch.tensor(segment_ids, dtype=torch.long).unsqueeze(0)
        baseline_ids = baseline_ids.to(device)
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)

        # record real input length
        input_len = int(input_mask[0].sum())

        # record [MASK]'s position
        tgt_pos = tokens_info["tokens"].index("[MASK]")

        # record various results
        res_dict = {"pred": [], "ig_pred": [], "ig_gold": [], "base": []}

        # original pred prob
        outputs = model(
            input_ids=input_ids,
            attention_mask=input_mask,
            token_type_ids=segment_ids,
        )
        logits = outputs.logits
        pred_label = int(torch.argmax(logits[0, tgt_pos, :]))  # scalar
        model.forward_with_partitioning(target_position=tgt_pos)
        gold_label = tokenizer.convert_tokens_to_ids(tokens_info["gold_obj"])
        tokens_info["pred_obj"] = tokenizer.convert_ids_to_tokens(pred_label)
        ig_gold = model.calulate_integrated_gradients(target_label=gold_label)
        for ig in ig_gold:
            ig = ig.cpu().detach()
            res_dict["ig_gold"].append(ig)

        if args.get_ig_gold:
            res_dict["ig_gold"] = convert_to_triplet_ig_top(
                res_dict["ig_gold"], args.retention_threshold
            )
            # res_dict['ig_gold'] = convert_to_triplet_ig(res_dict['ig_gold'])
        # if args.get_base:
        #     res_dict["base"] = convert_to_triplet_ig(res_dict["base"])
        # res_dict_bag.append([tokens_info, res_dict])
        res_dict_bag.append([res_dict])
        # record running time
        toc = time.perf_counter()
        print(f"***** Costing time: {toc - tic:0.4f} seconds *****")
        pprint(torch.cuda.memory_stats())
        model.clean()

    with jsonlines.open(os.path.join(args.output_dir, args.result_file), "w") as fw:
        fw.write(res_dict_bag)

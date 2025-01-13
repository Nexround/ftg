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
            max_length=512,
        )

    dataset = dataset["train"].shuffle(seed=42).select(range(args.num_sample))
    tokenized_train = dataset.map(tokenize_function, batched=True, num_proc=32)
    # evaluate args.debug bags for each relation

    res_dict_bag = []

    for item in tqdm(tokenized_train):
        # record running time
        tic = time.perf_counter()

        cls_id = tokenizer.convert_tokens_to_ids('[CLS]')
        # record various results
        res_dict = {"pred": [], "ig_pred": [], "ig_gold": [], "base": []}

        # Move input tensors to the same device as the model
        input_ids = torch.tensor(item["input_ids"]).unsqueeze(0).to(device)
        attention_mask = torch.tensor(item["attention_mask"]).unsqueeze(0).to(device)

        cls_pos = 0
        # cls_pos = input_ids.index(cls_id)

        # original pred prob
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        logits = outputs.logits
        predicted_class = int(torch.argmax(logits, dim=-1))  # 预测类别

        # predicted_class = torch.argmax(logits, dim=-1).item()
        print(f"Predicted class: {predicted_class}")
        label_map = model.config.id2label
        print(f"Predicted label: {label_map[predicted_class]}")
        model.forward_with_partitioning(target_position=cls_pos)
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
        # pprint(torch.cuda.memory_stats())
        model.clean()

    with jsonlines.open(os.path.join(args.output_dir, args.result_file), "w") as fw:
        fw.write(res_dict_bag)

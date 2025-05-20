from beir import util
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.lexical import BM25Search as BM25
from beir.reranking.models import CrossEncoder
from beir.reranking import Rerank
import os
from typing import List, Dict, Tuple
from FlagEmbedding import FlagReranker
from vllm import LLM, SamplingParams
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from tqdm.auto import trange
import pandas as pd
import time


def prep_bier_data(dataset_name):
    print(dataset_name + "\n")
    out_dir = os.path.join("/content/drive/MyDrive/", "bier_datasets")

    if "/" in dataset_name:
        data_path = os.path.join(out_dir, dataset_name)
    else:
        #### Download trec-covid.zip dataset and unzip the dataset
        url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(
            dataset_name
        )
        try:
            data_path = util.download_and_unzip(url, out_dir)
        except Exception as e:
            raise Exception(f"Failed to download dataset {dataset_name}: {e}")

    #### Provide the data path where trec-covid has been downloaded and unzipped to the data loader
    # data folder would contain these files:
    # (1) trec-covid/corpus.jsonl  (format: jsonlines)
    # (2) trec-covid/queries.jsonl (format: jsonlines)
    # (3) trec-covid/qrels/test.tsv (format: tsv ("\t"))
    return GenericDataLoader(data_path).load(split="test")


def get_bm_25_results_retriever(dataset_name, corpus, queries):
    #########################################
    #### (1) RETRIEVE Top-100 docs using BM25
    #########################################

    #### Provide parameters for Elasticsearch
    hostname = "localhost"  # localhost
    index_name = dataset_name.replace("/", "__")
    initialize = False  # False

    language = (
        "cjk"
        if any([x in dataset_name for x in ["chinese", "japanese", "korean"]])
        else "english"
    )
    model = BM25(
        index_name=index_name,
        hostname=hostname,
        initialize=initialize,
        language=language,
    )
    retriever = EvaluateRetrieval(model)

    #### Retrieve dense results (format of results is identical to qrels)
    results = retriever.retrieve(corpus, queries)

    return results, retriever


def get_cross_encoder_results(
    cross_encoder_model, corpus, queries, qrels, results, retriever
):
    ################################################
    #### (2) RERANK Top-100 docs using Cross-Encoder
    ################################################
    reranker = Rerank(cross_encoder_model, batch_size=128)

    print(len(queries))
    # Rerank top-100 results using the reranker provided
    rerank_results = reranker.rerank(corpus, queries, results, top_k=100)

    #### Evaluate your retrieval using NDCG@k, MAP@K ...
    return EvaluateRetrieval.evaluate(qrels, rerank_results, retriever.k_values)


class FlagEmbeddingCrossEncoder:
    def __init__(self, model_name: str, **kwargs):
        self.reranker = FlagReranker(model_name, use_fp16=True)

    def predict(
        self,
        sentences: List[Tuple[str, str]],
        batch_size: int = 32,
        show_progress_bar: bool = True,
    ) -> List[float]:
        return self.reranker.compute_score(sentences, batch_size=batch_size)


class LBCrossEncoder:
    def __init__(self, model_path: str, **kwargs):
        self.llm = LLM(model=model_path, gpu_memory_utilization=0.5)
        self.sampling_params = SamplingParams(
            temperature=0.0, logprobs=14, max_tokens=1
        )
        self.tok = self.llm.llm_engine.tokenizer.tokenizer
        self.idx_tokens = [self.tok.encode(str(i))[0] for i in range(1, 8)]

    def make_reranker_input(self, t, q):
        return f"<<<Query>>>\n{q}\n\n<<<Context>>>\n{t}"

    def make_reranker_training_datum(self, context, question):
        system_message = "Given a query and a piece of text, output a score of 1-7 based on how related the query is to the text. 1 means least related and 7 is most related."

        return [
            {"role": "system", "content": system_message},
            {"role": "user", "content": self.make_reranker_input(context, question)},
        ]

    def get_prob(self, logprob_dict, tok_id):
        return (
            np.exp(logprob_dict[tok_id].logprob) if tok_id in logprob_dict.keys() else 0
        )

    def predict(
        self,
        sentences: List[Tuple[str, str]],
        batch_size: int = 32,
        show_progress_bar: bool = True,
    ) -> List[float]:
        chats = [self.make_reranker_training_datum(c, q) for q, c in sentences]
        responses = self.llm.chat(chats, self.sampling_params)
        probs = np.array(
            [
                [self.get_prob(r.outputs[0].logprobs[0], y) for y in self.idx_tokens]
                for r in responses
            ]
        )
        scores = probs[:, 3:].mean(axis=1) - probs[:, :4].mean(axis=1)
        return scores


class AlibabaCrossEncoder:
    def __init__(self, model_path: str, **kwargs):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path, trust_remote_code=True, torch_dtype=torch.float16
        ).to(torch.device("cuda"))
        self.model.eval()

    def predict(
        self,
        sentences: List[Tuple[str, str]],
        batch_size: int = 32,
        show_progress_bar: bool = True,
    ) -> List[float]:
        range_fn = trange if show_progress_bar else range
        scores = []
        with torch.no_grad():
            for i in range_fn(0, len(sentences), batch_size):
                inputs = self.tokenizer(
                    sentences[i : i + batch_size],
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                    max_length=512,
                ).to(torch.device("cuda"))
                scores.extend(
                    self.model(**inputs, return_dict=True)
                    .logits.view(
                        -1,
                    )
                    .cpu()
                    .numpy()
                    .tolist()
                )
        return scores


dataset_names = [
    "arguana",
    "dbpedia-entity",
    "fiqa",
    "nfcorpus",
    "scidocs",
    "scifact",
    "trec-covid-v2",
    "vihealthqa",
    "webis-touche2020",
]

lightbl_ce, lightbl_name = (
    LBCrossEncoder("/cache/models/LoKI_models/loki_30_a_0512_1_real"),
    "loki_30_a_0512_1",
)


models = [
    # (flagemb_ce, flagemb_name),
    (lightbl_ce, lightbl_name),
    # (sentemb_ce, sentemb_name),
    # (alibaba_ce, alibaba_name)
]


dataset_names = [
    "arguana",
    "dbpedia-entity",
    "fiqa",
    "nfcorpus",
    "scidocs",
    "scifact",
    "trec-covid-v2",
    "vihealthqa",
    "webis-touche2020",
]

for dataset_name in dataset_names:
    print(dataset_name)

    corpus, queries, qrels = prep_bier_data(dataset_name)

    # Select first 250 queries, sorted by query key, to save computation time
    queries = {x: queries[x] for x in sorted(queries.keys())[:250]}
    results, retriever = get_bm_25_results_retriever(dataset_name, corpus, queries)

    for cross_encoder_model, model_name in models:
        print(model_name)

        t0 = time.time()
        ndcg, _map, recall, precision = get_cross_encoder_results(
            cross_encoder_model, corpus, queries, qrels, results, retriever
        )
        time_elapsed = time.time() - t0

        save_dir = f"./bier_results/" + model_name.replace("/", "__")
        os.makedirs(save_dir, exist_ok=True)

        save_path = f"{save_dir}/" + dataset_name.replace("/", "__") + ".parquet"

        pd.DataFrame(
            dict(**ndcg, **_map, **recall, **precision, time=time_elapsed),
            index=[[dataset_name, model_name]],
        ).to_parquet(save_path)

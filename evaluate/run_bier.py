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
    initialize = True  # False

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
    LBCrossEncoder("/cache/models/loki_reranker_qwen2_5-0-5b-20_real"),
    "loki_reranker_qwen2_5-0-5b-20_real",
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
# from beir import util
# from beir.datasets.data_loader import GenericDataLoader
# from beir.retrieval.evaluation import EvaluateRetrieval
# from beir.retrieval.search.lexical import BM25Search as BM25
# from beir.reranking import Rerank
# import os
# import pickle
# from typing import List, Dict, Tuple
# from vllm import LLM, SamplingParams
# import numpy as np
# import torch
# from tqdm.auto import trange
# import pandas as pd
# import time

# # 配置常量
# CACHE_ROOT = "/content/drive/MyDrive/bier_cache"
# BM25_CACHE_DIR = os.path.join(CACHE_ROOT, "bm25_results")
# DATASET_CACHE_DIR = os.path.join(CACHE_ROOT, "datasets")
# RESULTS_DIR = os.path.join(CACHE_ROOT, "evaluation_results")
# os.makedirs(BM25_CACHE_DIR, exist_ok=True)
# os.makedirs(DATASET_CACHE_DIR, exist_ok=True)
# os.makedirs(RESULTS_DIR, exist_ok=True)

# def prep_bier_data(dataset_name: str) -> Tuple[Dict, Dict, Dict]:
#     """加载数据集并缓存"""
#     dataset_dir = os.path.join(DATASET_CACHE_DIR, dataset_name.replace("/", "__"))
    
#     if not os.path.exists(dataset_dir):
#         if "/" in dataset_name:  # 本地数据集
#             os.symlink(dataset_name, dataset_dir)
#         else:  # 下载数据集
#             print(f"Downloading dataset: {dataset_name}")
#             url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset_name}.zip"
#             util.download_and_unzip(url, DATASET_CACHE_DIR)
#             os.rename(os.path.join(DATASET_CACHE_DIR, dataset_name), dataset_dir)
    
#     return GenericDataLoader(dataset_dir).load(split="test")

# def get_bm25_results(dataset_name: str, corpus: Dict, queries: Dict) -> Tuple[Dict, object]:
#     """获取BM25结果并缓存"""
#     cache_path = os.path.join(BM25_CACHE_DIR, f"{dataset_name.replace('/','__')}.pkl")
    
#     if os.path.exists(cache_path):
#         with open(cache_path, "rb") as f:
#             return pickle.load(f)
    
#     # 初始化BM25引擎
#     model = BM25(
#         index_name=dataset_name.replace("/", "__"),
#         hostname="localhost",
#         initialize=True,
#         language="cjk" if any(x in dataset_name for x in ["chinese", "japanese", "korean"]) else "english"
#     )
#     retriever = EvaluateRetrieval(model)
#     results = retriever.retrieve(corpus, queries)
    
#     # 缓存结果
#     with open(cache_path, "wb") as f:
#         pickle.dump((results, retriever), f)
    
#     return results, retriever

# class LBCrossEncoder:
#     """优化后的重排序模型类"""
#     def __init__(self, model_path: str):
#         self.llm = LLM(model=model_path, gpu_memory_utilization=0.5)
#         self.sampling_params = SamplingParams(temperature=0.0, logprobs=14, max_tokens=1)
#         self.tokenizer = self.llm.llm_engine.tokenizer.tokenizer
#         self.score_tokens = [self.tokenizer.encode(str(i))[0] for i in range(1, 8)]
    
#     def _create_input(self, context: str, query: str) -> List[Dict]:
#         return [{
#             "role": "system",
#             "content": "Given a query and context, rate relevance from 1-7 (1=least, 7=most)."
#         }, {
#             "role": "user",
#             "content": f"<<<Query>>>\n{query}\n\n<<<Context>>>\n{context}"
#         }]
    
#     def predict(self, sentence_pairs: List[Tuple[str, str]], batch_size: int = 32) -> List[float]:
#         scores = []
#         for i in trange(0, len(sentence_pairs), batch_size):
#             batch = sentence_pairs[i:i+batch_size]
#             chats = [self._create_input(c, q) for q, c in batch]
#             responses = self.llm.chat(chats, self.sampling_params)
            
#             batch_scores = []
#             for res in responses:
#                 logprobs = res.outputs[0].logprobs[0]
#                 probs = [np.exp(logprobs[t].logprob) if t in logprobs else 0 for t in self.score_tokens]
#                 batch_scores.append(np.mean(probs[3:]) - np.mean(probs[:3]))  # 4-7分 vs 1-3分
#             scores.extend(batch_scores)
#         return scores

# def evaluate_model(
#     model: object,
#     model_name: str,
#     dataset_name: str,
#     corpus: Dict,
#     queries: Dict,
#     qrels: Dict,
#     results: Dict
# ) -> pd.DataFrame:
#     """评估模型并缓存结果"""
#     result_path = os.path.join(RESULTS_DIR, f"{dataset_name.replace('/','__')}_{model_name}.parquet")
    
#     if os.path.exists(result_path):
#         print(f"Results exist, skipping {model_name} on {dataset_name}")
#         return pd.read_parquet(result_path)
    
#     # 重排序和评估
#     reranker = Rerank(model, batch_size=128)
#     start_time = time.time()
#     rerank_results = reranker.rerank(corpus, queries, results, top_k=100)
#     eval_metrics = EvaluateRetrieval.evaluate(qrels, rerank_results, [10, 100])
    
#     # 构建结果DataFrame
#     metrics = {
#         **eval_metrics["ndcg"],
#         **eval_metrics["map"],
#         **eval_metrics["recall"],
#         **eval_metrics["precision"],
#         "time": time.time() - start_time
#     }
#     result_df = pd.DataFrame(metrics, index=[f"{dataset_name}|{model_name}"])
    
#     # 保存结果
#     result_df.to_parquet(result_path)
#     return result_df

# def main():
#     # 配置实验参数
#     datasets = [
#         "arguana", "dbpedia-entity", "fiqa", "nfcorpus",
#         "scidocs", "scifact", "trec-covid-v2", "vihealthqa", "webis-touche2020"
#     ]
#     models = [
#         (LBCrossEncoder("/cache/models/loki_reranker_qwen2_5-0-5b-20_real"), "loki_reranker")
#     ]
    
#     # 主执行循环
#     for dataset in datasets:
#         print(f"\nProcessing dataset: {dataset}")
        
#         # 加载数据
#         corpus, queries, qrels = prep_bier_data(dataset)
#         queries = {k: queries[k] for k in sorted(queries)[:250]}  # 取前250个查询
        
#         # 获取BM25结果
#         bm25_results, retriever = get_bm25_results(dataset, corpus, queries)
        
#         # 评估所有模型
#         for model, model_name in models:
#             evaluate_model(model, model_name, dataset, corpus, queries, qrels, bm25_results)

# if __name__ == "__main__":
#     main()
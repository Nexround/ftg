echo '{
  "reranker_continuous_filt_max7_train": {
    "hf_hub_url": "/cache/huggingface/datasets/lightblue___reranker_continuous_filt_max7_train/default/0.0.0/32bd1c820d8da33f7f4780ae2252f1d3b82377f0",
    "formatting": "sharegpt"
  }
}' > /workspace/LLaMA-Factory/data/dataset_info.json

cd /workspace/LLaMA-Factory/ && llamafactory-cli train /workspace/ftg/lf_train/Qwen2.5-0.5B-Instruct/reranker.yaml

# rm -r /root/train_outputs/Qwen2.5-0.5B-Instruct/reranker_continuous_filt_max7_train/checkpoint*
# huggingface-cli upload lightblue/reranker_0.5_cont_filt_7max /root/train_outputs/Qwen2.5-0.5B-Instruct/reranker_continuous_filt_max7_train
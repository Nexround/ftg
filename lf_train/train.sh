echo '{
  "reranker_continuous_filt_max7_train": {
    "hf_hub_url": "lightblue/reranker_continuous_filt_max7_train",
    "formatting": "sharegpt"
  }
}' > /workspace/LLaMA-Factory/data/dataset_info.json

cd /workspace/LLaMA-Factory/ && FORCE_TORCHRUN=1 llamafactory-cli train /workspace/ftg/lf_train/reranker.yaml

# rm -r /root/train_outputs/Qwen2.5-0.5B-Instruct/reranker_continuous_filt_max7_train/checkpoint*
# huggingface-cli upload lightblue/reranker_0.5_cont_filt_7max /root/train_outputs/Qwen2.5-0.5B-Instruct/reranker_continuous_filt_max7_train
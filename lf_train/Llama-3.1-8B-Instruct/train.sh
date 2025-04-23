echo '{
  "ToolACE": {
    "hf_hub_url": "/workspace/ToolACE",
    "formatting": "sharegpt",
    "columns": {
      "messages": "conversations",
      "system": "system"
  }
  }
}' > /workspace/LLaMA-Factory/data/dataset_info.json

cd /workspace/LLaMA-Factory/ && llamafactory-cli train /workspace/ftg/lf_train/8B/toolace.yaml

# rm -r /root/train_outputs/Qwen2.5-0.5B-Instruct/reranker_continuous_filt_max7_train/checkpoint*
# huggingface-cli upload lightblue/reranker_0.5_cont_filt_7max /root/train_outputs/Qwen2.5-0.5B-Instruct/reranker_continuous_filt_max7_train
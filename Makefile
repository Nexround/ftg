wiki:
	python ./src/data_inference.py \
	--bert_model bert-base-cased \
	--output_dir ./results/ \
	--gpus 0 \
	--max_seq_length 128 \
	--get_ig_gold \
	--batch_size 20 \
	--num_batch 1 \
	--debug 100000 \
	--num_sample 10000 \
	--result_file wiki_o.jsonl \
	--dataset "wikitext,wikitext-2-raw-v1" \
	--retention_threshold 98
wiki_legacy:
	python ./src/data_inference.py \
	--bert_model bert-base-cased \
	--output_dir ./results/ \
	--gpus 0 \
	--max_seq_length 128 \
	--get_ig_gold \
	--batch_size 20 \
	--num_batch 1 \
	--debug 100000 \
	--num_sample 10000 \
	--result_file wiki_o.jsonl \
	--dataset "wikipedia,20220301.en" \
	--retention_threshold 98
law:
	python ./src/data_inference.py \
	--bert_model bert-base-cased \
	--output_dir ./results/ \
	--gpus 0 \
	--max_seq_length 128 \
	--get_ig_gold \
	--batch_size 20 \
	--num_batch 1 \
	--debug 100000 \
	--num_sample 10000 \
	--result_file law_o.jsonl \
	--dataset "free-law/Caselaw_Access_Project" \
	--retention_threshold 98
imdb:
	python ./src/my_inference.py \
	--bert_model bert-base-cased \
	--output_dir ./results/ \
	--gpus 0 \
	--max_seq_length 128 \
	--get_ig_gold \
	--batch_size 20 \
	--num_batch 1 \
	--debug 100000 \
	--num_sample 10000 \
	--result_file imdb.json \
	--dataset imdb

token_cls_imdb:
	python ./src/token_cls.py \
	--bert_model /openbayes/home/ftg/results/train_full_imdb \
	--output_dir ./results \
	--max_seq_length 256 \
	--batch_size 20 \
	--result_file token_cls_imdb.json \
	--dataset imdb \
	--num_sample 10000 \
	--retention_threshold 99 

lm_h4:
	python lm_analyse.py \
	--model_path Qwen/Qwen2.5-0.5B-Instruct \
	--output_dir ./results \
	--max_seq_length 32768 \
	--times 8 \
	--result_file lm_h4.json \
	--dataset HuggingFaceH4/helpful-instructions \
	--num_sample 10000 \
	--retention_threshold 99 
	
lm_trivia_qa:
	python lm_analyse.py \
	--model_path Qwen/Qwen2.5-0.5B-Instruct \
	--output_dir ./results \
	--max_seq_length 32768 \
	--times 7 \
	--result_file lm_trivia_qa.json \
	--dataset mandarjoshi/trivia_qa,rc.nocontext \
	--num_sample 10000 \
	--retention_threshold 99 

lm_mmlu:
	accelerate launch \
		--mixed_precision bf16 \
		mmlu_analyse.py \
		--model_path Qwen/Qwen2.5-0.5B-Instruct \
		--output_dir ./target_neurons \
		--max_seq_length 32768 \
		--times 7 \
		--result_file lm_mmlu.json \
		--percentage 90 \
		--write_mode w

lm_mmlu_hdf5:
	accelerate launch \
		--mixed_precision bf16 \
		mmlu_analyse_hdf5.py \
		--model_path Qwen/Qwen2.5-0.5B-Instruct \
		--output_dir ./hdf5 \
		--max_seq_length 32768 \
		--times 7 \
		--result_file mmlu.h5 \
		--write_mode w

lm_mmlu_qwen7b_hdf5:
	accelerate launch \
		--mixed_precision bf16 \
		mmlu_analyse_hdf5.py \
		--model_path Qwen/Qwen2.5-7B-Instruct \
		--output_dir ./hdf5 \
		--max_seq_length 32768 \
		--times 7 \
		--result_file qwen7b_mmlu.h5 \
		--write_mode w

token_cls_agnews:
	python ./src/token_cls.py \
	--bert_model /openbayes/home/ftg/results/agnews_checkpoint-22500 \
	--output_dir ./results \
	--max_seq_length 256 \
	--batch_size 20 \
	--result_file token_cls_agnews.json \
	--dataset fancyzhx/ag_news \
	--num_sample 10000 \
	--retention_threshold 99 

token_cls_agnews_bert_imdb_cls:
	python ./src/token_cls.py \
	--bert_model /openbayes/home/ftg/results/agnews_checkpoint-22500 \
	--output_dir ./results \
	--max_seq_length 128 \
	--batch_size 20 \
	--result_file token_cls_agnews_bert_imdb_cls.json \
	--dataset imdb \
	--num_sample 10000 \
	--retention_threshold 99 

tokencls_agnews:
	python ./src/token_cls.py \
	--bert_model /root/ftg/results/new_6tags_agnews_based_imdb \
	--output_dir ./results \
	--max_seq_length 256 \
	--batch_size 20 \
	--result_file tokencls_agnews.json \
	--dataset fancyzhx/ag_news \
	--num_sample 10000 \
	--retention_threshold 99 

tokencls_agnews_on_imdb:
	python ./src/token_cls.py \
	--bert_model /root/ftg/results/new_6tags_agnews_based_imdb \
	--output_dir ./results \
	--max_seq_length 256 \
	--batch_size 20 \
	--result_file tokencls_agnews_on_imdb.json \
	--dataset imdb \
	--num_sample 10000 \
	--retention_threshold 99 

train_full_yelp:
	python train.py \
	--dataset "Yelp/yelp_review_full" \
	--model "distilbert/distilbert-base-uncased" \
	--output_dir "/root/ftg/results" \
	--output_prefix "experiment_name" \
	--batch_size 32 \
	--num_labels 5 \
	--learning_rate 5e-5 \
	--full 

train_full_ag_news:
	python train.py \
	--dataset "fancyzhx/ag_news" \
	--model "distilbert/distilbert-base-uncased" \
	--output_dir "./results" \
	--output_prefix "experiment_name" \
	--batch_size 16 \
	--num_labels 5 \
	--learning_rate 5e-5 \
	--full \
	--num_train_epochs 3

train_full_ag_news_bert:
# 原始bert
	python train.py \
	--dataset "fancyzhx/ag_news" \
	--model "bert-base-cased" \
	--output_dir "./results" \
	--output_prefix "experiment_name" \
	--batch_size 16 \
	--num_labels 4 \
	--learning_rate 5e-5 \
	--full \
	--num_train_epochs 3

train_full_ag_news_bert_6:
# 原始bert 0.9468421052631579
	python train.py \
	--dataset "fancyzhx/ag_news" \
	--model "bert-base-cased" \
	--output_dir "./results" \
	--output_prefix "experiment_name" \
	--batch_size 16 \
	--num_labels 6 \
	--learning_rate 5e-5 \
	--full \
	--num_train_epochs 3

train_full_amazon:
	python train.py \
	--dataset "yassiracharki/Amazon_Reviews_for_Sentiment_Analysis_fine_grained_5_classes" \
	--model "distilbert/distilbert-base-uncased" \
	--output_dir "/root/ftg/results" \
	--output_prefix "experiment_name" \
	--batch_size 64 \
	--num_labels 6 \
	--learning_rate 5e-5 \
	--full \
	--num_train_epochs 6

train_full_imdb:
	python train.py \
	--dataset "imdb" \
	--model "bert-base-cased" \
	--output_dir "/root/ftg/results" \
	--output_prefix "experiment_name" \
	--batch_size 16 \
	--num_labels 2 \
	--full

train_full_imdb_on_agnews:
# 在agnews模型基础上训练imdb
# 'eval_accuracy': 0.92044
	python train.py \
	--dataset "imdb" \
	--model "/openbayes/home/ftg/results/agnews_checkpoint-22500" \
	--output_dir "./results" \
	--output_prefix "experiment_name" \
	--batch_size 16 \
	--num_labels 2 \
	--full
	
train_target:
	python train.py \
	--dataset "imdb" \
	--target_neurons_path "/root/ftg/src/train/target_neurons/complement_1.json" \
	--model "bert-base-uncased" \
	--output_dir "/root/ftg/results" \
	--output_prefix "experiment_name" \
	--batch_size 16 \
	--num_labels 2 \
	--train_target_neurons \

train_target_sst2:
	python train.py \
	--dataset "nyu-mll/glue,sst2" \
	--target_neurons_path "/root/ftg/src/train/target_neurons/complement_1.json" \
	--model "bert-base-uncased" \
	--output_dir "/root/ftg/results" \
	--output_prefix "experiment_name" \
	--batch_size 16 \
	--num_labels 2 \
	--train_target_neurons \

train_target_token_cls_agnews_bert_imdb_cls_complement_1:
	python train.py \
	--dataset "imdb" \
	--target_neurons_path "/openbayes/home/ftg/token_cls_agnews_bert_imdb_cls_complement_1.json" \
	--model "/openbayes/home/ftg/results/agnews_checkpoint-22500" \
	--output_dir "/root/ftg/results" \
	--output_prefix "experiment_name" \
	--batch_size 16 \
	--num_labels 2 \
	--train_target_neurons \

train_target_token_cls_agnews_bert_imdb_cls_complement_1_high:
# 'eval_accuracy': 0.90916
	python train.py \
	--dataset "imdb" \
	--target_neurons_path "/openbayes/home/ftg/token_cls_agnews_bert_imdb_cls_complement_1.json" \
	--model "/openbayes/home/ftg/results/agnews_checkpoint-22500" \
	--output_dir "./results" \
	--output_prefix "experiment_name" \
	--batch_size 16 \
	--num_labels 2 \
	--train_target_neurons \
	--learning_rate 5e-4 \
	
train_target_token_cls_agnews_bert_imdb_cls_complement_1_high_improved:
	python train_target.py \
	--dataset "imdb" \
	--target_neurons_path "/root/ftg/token_cls_agnews_bert_imdb_cls_complement_1.json" \
	--model "/root/ftg/results/new_6tags_agnews_based_imdb" \
	--output_dir "./results" \
	--output_prefix "experiment_name" \
	--batch_size 16 \
	--num_labels 6 \
	--train_target_neurons \
	--learning_rate 1e-4 \

train_target_token_cls_agnews_bert_imdb_random:
	python train_target.py \
	--dataset "imdb" \
	--target_neurons_path "/root/ftg/random_data.json" \
	--model "/root/ftg/results/new_6tags_agnews_based_imdb" \
	--output_dir "./results" \
	--output_prefix "experiment_name" \
	--batch_size 16 \
	--num_labels 6 \
	--train_target_neurons \
	--learning_rate 1e-5 \

train_target_token_cls_agnews_bert_imdb_6:
	python train_target.py \
	--dataset "imdb" \
	--target_neurons_path "/root/ftg/token_cls_agnews_bert_imdb_cls_complement_1.json" \
	--model "/root/ftg/results/new_6tags_agnews_based_imdb" \
	--output_dir "./results" \
	--output_prefix "experiment_name" \
	--batch_size 16 \
	--num_labels 6 \
	--full \
	--learning_rate 5e-5 \

train_target_0115agnews_imdb:
	python train_target.py \
	--dataset "imdb" \
	--target_neurons_path "/root/ftg/0115agnews_imdb.json" \
	--model "/root/ftg/results/new_6tags_agnews_based_imdb" \
	--output_dir "./results" \
	--output_prefix "experiment_name" \
	--batch_size 16 \
	--num_labels 6 \
	--train_target_neurons \
	--learning_rate 5e-5 \
	
train_target_0115agnews_imdb_random:
	python train_target.py \
	--dataset "imdb" \
	--target_neurons_path "/root/ftg/random_data_1537.json" \
	--model "/root/ftg/results/new_6tags_agnews_based_imdb" \
	--output_dir "./results" \
	--output_prefix "experiment_name" \
	--batch_size 16 \
	--num_labels 6 \
	--train_target_neurons \
	--learning_rate 5e-5 \

train_target_0115agnews_imdb:
	python train_target.py \
	--dataset "imdb" \
	--target_neurons_path "/root/ftg/complement_2.json" \
	--model "/root/ftg/results/new_6tags_agnews_based_imdb" \
	--output_dir "./results" \
	--output_prefix "experiment_name" \
	--batch_size 16 \
	--num_labels 6 \
	--train_target_neurons \
	--learning_rate 5e-5 \
	
train_target_0115agnews_imdb_random:
	python train_target.py \
	--dataset "imdb" \
	--target_neurons_path "/root/ftg/random_data_1537.json" \
	--model "/root/ftg/results/new_6tags_agnews_based_imdb" \
	--output_dir "./results" \
	--output_prefix "experiment_name" \
	--batch_size 16 \
	--num_labels 6 \
	--train_target_neurons \
	--learning_rate 5e-5 \

train_target_0115agnews_imdb_random_5247:
	python train_target.py \
	--dataset "imdb" \
	--target_neurons_path "/root/ftg/random_data_5247.json" \
	--model "/root/ftg/results/new_6tags_agnews_based_imdb" \
	--output_dir "./results" \
	--output_prefix "experiment_name" \
	--batch_size 16 \
	--num_labels 6 \
	--train_target_neurons \
	--learning_rate 5e-5 \

train_target_0115agnews_imdb_random_10:
	python train_target.py \
	--dataset "imdb" \
	--target_neurons_path "/root/ftg/random_data_10.json" \
	--model "/root/ftg/results/new_6tags_agnews_based_imdb" \
	--output_dir "./results" \
	--output_prefix "experiment_name" \
	--batch_size 16 \
	--num_labels 6 \
	--train_target_neurons \
	--learning_rate 5e-5 \

train_ffn:
	python train_target.py \
	--dataset "imdb" \
	--model "/root/ftg/results/new_6tags_agnews_based_imdb" \
	--output_dir "./results" \
	--output_prefix "experiment_name" \
	--batch_size 16 \
	--num_labels 6 \
	--ffn \
	--learning_rate 5e-5 \

train_lora_imdb:
	python train_target.py \
	--dataset "imdb" \
	--model "/root/ftg/results/new_6tags_agnews_based_imdb" \
	--output_dir "./results" \
	--output_prefix "experiment_name" \
	--batch_size 16 \
	--num_labels 6 \
	--train_option lora \
	--learning_rate 5e-5 \

train_scinews_target:
	python train_target.py \
	--dataset "/cache/huggingface/datasets/SciNews_3labels" \
	--target_neurons_path "/root/ftg/target_neurons/0115agnews_imdb.json" \
	--model "/root/ftg/results/new_6tags_agnews_based_imdb" \
	--output_dir "./results" \
	--output_prefix "experiment_name" \
	--batch_size 16 \
	--num_labels 6 \
	--train_option target_neurons \
	--learning_rate 5e-5 \

train_scinews_target_based_on_agnews_3labels:
	python train_target.py \
	--dataset "/cache/huggingface/datasets/SciNews_3labels" \
	--target_neurons_path "/root/ftg/target_neurons/0115agnews_imdb.json" \
	--model "/root/ftg/results/_cache_huggingface_datasets_agnews_3labels_01_18_12:23/checkpoint-16875" \
	--output_dir "./results" \
	--output_prefix "experiment_name" \
	--batch_size 16 \
	--num_labels 6 \
	--train_option target_neurons \
	--learning_rate 5e-5 \

train_scinews_target_based_on_agnews_3labels_random:
	python train_target.py \
	--dataset "/cache/huggingface/datasets/SciNews_3labels" \
	--target_neurons_path "/root/ftg/target_neurons/random_data_5247.json" \
	--model "/root/ftg/results/_cache_huggingface_datasets_agnews_3labels_01_18_12:23/checkpoint-16875" \
	--output_dir "./results" \
	--output_prefix "experiment_name" \
	--batch_size 16 \
	--num_labels 6 \
	--train_option target_neurons \
	--learning_rate 5e-5 \

train_scinews_target_based_on_agnews_3labels_random_10:
	python train_target.py \
	--dataset "/cache/huggingface/datasets/SciNews_3labels" \
	--target_neurons_path "/root/ftg/target_neurons/random_data_10.json" \
	--model "/root/ftg/results/agnews_3labels_01_18_12:23/checkpoint-16875" \
	--output_dir "./results" \
	--output_prefix "experiment_name" \
	--batch_size 16 \
	--num_labels 6 \
	--train_option target_neurons \
	--learning_rate 5e-5 \

train_scinews_target_based_on_agnews_3labels_random_100:
	python train_target.py \
	--dataset "/cache/huggingface/datasets/SciNews_3labels" \
	--target_neurons_path "/root/ftg/target_neurons/random_data_100.json" \
	--model "/root/ftg/results/agnews_3labels_01_18_12:23/checkpoint-16875" \
	--output_dir "./results" \
	--output_prefix "experiment_name" \
	--batch_size 16 \
	--num_labels 6 \
	--train_option target_neurons \
	--learning_rate 5e-5 \

train_agnews_6_full:
	python train_target.py \
	--dataset "/cache/huggingface/datasets/agnews_3labels" \
	--model "bert-base-cased" \
	--output_dir "./results" \
	--output_prefix "experiment_name" \
	--batch_size 16 \
	--num_labels 6 \
	--train_option full \
	--learning_rate 5e-5 \

train_agnews_6_full_scinews_c_full:
	python train_target.py \
	--dataset "/cache/huggingface/datasets/SciNews_3labels" \
	--model "/root/ftg/results/agnews_3labels_01_18_12:23/checkpoint-16875" \
	--output_dir "./results" \
	--output_prefix "experiment_name" \
	--batch_size 16 \
	--num_labels 6 \
	--train_option c_full \
	--learning_rate 5e-5 \

train_agnews_6_full_scinews_ffn:
	python train_target.py \
	--dataset "/cache/huggingface/datasets/SciNews_3labels" \
	--model "/root/ftg/results/agnews_3labels_01_18_12:23/checkpoint-16875" \
	--output_dir "./results" \
	--output_prefix "experiment_name" \
	--batch_size 16 \
	--num_labels 6 \
	--train_option ffn \
	--learning_rate 5e-5 \
	
train_agnews_6_full_scinews_lora:
	python train_target.py \
	--dataset "/cache/huggingface/datasets/SciNews_3labels" \
	--model "/root/ftg/results/agnews_3labels_01_18_12:23/checkpoint-16875" \
	--output_dir "./results" \
	--output_prefix "experiment_name" \
	--batch_size 16 \
	--num_labels 6 \
	--train_option lora \
	--learning_rate 5e-5 \
	
train_agnews_14_full:
	python train_target.py \
	--dataset "/cache/huggingface/datasets/ag_news_14labels" \
	--label_json /root/ftg/train_config/label_0.json \
	--model "bert-base-cased" \
	--output_dir "./results" \
	--output_prefix "experiment_name" \
	--batch_size 32 \
	--num_labels 22 \
	--train_option full \
	--learning_rate 5e-5 \

train_agnews_14_full_roberta:
	python train_target.py \
	--dataset "/cache/huggingface/datasets/ag_news_14labels" \
	--label_json /root/ftg/train_config/label_0.json \
	--model "FacebookAI/roberta-base" \
	--output_dir "./results" \
	--output_prefix "experiment_name" \
	--batch_size 32 \
	--num_labels 22 \
	--train_option full \
	--learning_rate 5e-5 \
	--num_train_epochs 9

train_qwen_reranker_loki:
	accelerate launch train_qwen.py \
	--dataset "lightblue/reranker_continuous_filt_max7_train" \
	--target_neurons_path "target_neurons/selected_neurons_spindle.json" \
	--model "Qwen/Qwen2.5-0.5B-Instruct" \
	--output_dir "./results" \
	--output_prefix "experiment_name" \
	--batch_size 1 \
	--num_train_epochs 1 \
	--train_option loki \
	--learning_rate 1e-5 \

train_qwen_reranker_lora:
	accelerate launch train_qwen.py \
	--dataset "../reranker_continuous_filt_max7_train" \
	--model "Qwen/Qwen2.5-0.5B-Instruct" \
	--output_dir "./results" \
	--output_prefix "experiment_name" \
	--batch_size 1 \
	--num_train_epochs 1 \
	--train_option lora \
	--learning_rate 1e-4 \
	--lora_rank 8 \
	--lora_alpha 16

train_qwen_reranker_ffn:
	accelerate launch train_qwen.py \
	--dataset "lightblue/reranker_continuous_filt_max7_train" \
	--model "Qwen/Qwen2.5-0.5B-Instruct" \
	--output_dir "./results" \
	--output_prefix "experiment_name" \
	--batch_size 1 \
	--num_train_epochs 1 \
	--train_option ffn \
	--learning_rate 1e-5 \
	--ds_config train_config/ds_config.json


train_qwen_reranker_full:
	python train_qwen.py \
	--dataset "/cache/huggingface/datasets/reranker_conversations_converted" \
	--model "Qwen/Qwen2.5-0.5B-Instruct" \
	--output_dir "./results" \
	--output_prefix "experiment_name" \
	--batch_size 1 \
	--num_train_epochs 1 \
	--train_option full \
	--learning_rate 1e-4 \

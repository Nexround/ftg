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
	--output_dir "/root/ftg/results" \
	--output_prefix "experiment_name" \
	--batch_size 16 \
	--num_labels 5 \
	--learning_rate 5e-5 \
	--full \
	--num_train_epochs 6

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

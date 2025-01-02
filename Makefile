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
	--result_file imdb.json \
	--dataset imdb
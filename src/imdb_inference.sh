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
--result_file ./results/imdb.json\
--dataset '("imdb",)'
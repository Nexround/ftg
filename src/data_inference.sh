python data_inference.py \
    --bert_model bert-base-cased \
    --data_path ../data/PARAREL/data_all.json \
    --tmp_data_path ../data/PARAREL/data_all_allbags.json \
    --output_dir ../results/ \
    --output_prefix TREx-all \
    --gpus 0 \
    --max_seq_length 128 \
    --get_ig_gold \
    --batch_size 20 \
    --num_batch 1 \
    --pt_relation $1 \
    --debug 100000 \
    --num_sample 100
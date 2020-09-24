cd "/home/theorist17/projects/adapter/adapter-transformers"
export DATA_DIR="/home/theorist17/projects/adapter/data"
export TASK_NAME=commonsenseqa

python3 run_cqa_bert.py \
--model_name_or_path bert-base-cased \
--task_name $TASK_NAME \
--do_train \
--do_eval \
--data_dir "$DATA_DIR/$TASK_NAME" \
--max_seq_length 128 \
--learning_rate 5e-5 \
--logging_steps 100 \
--num_train_epochs 10 \
--output_dir "home/theorist17/projects/adapter/adapters/$TASK_NAME" \
--per_gpu_eval_batch_size=16 \
--per_device_train_batch_size=16 \
--gradient_accumulation_steps 2 \
--overwrite_output \
--evaluate_during_training
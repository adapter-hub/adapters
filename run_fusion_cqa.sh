# REQUIRED: DOWNLOAD SST FROME GLUE $ python3 transformers/utils/download_glue_data.py --tasks MNLI
cd "/home/theorist17/projects/adapter/adapter-transformers"
export DATA_DIR="home/theorist17/projects/adapter/data"
export TASK_NAME=commonsenseqa

python3 run_fusion_cqa.py \
--model_name_or_path bert-base-cased \
--task_name $TASK_NAME \
--do_train \
--do_eval \
--data_dir "$DATA_DIR/$TASK_NAME" \
--max_seq_length 80 \
--learning_rate 5e-5 \
--num_train_epochs 3 \
--output_dir "home/theorist17/projects/adapter/fusions/$TASK_NAME" \
--per_gpu_eval_batch_size=16 \
--per_device_train_batch_size=16 \
--gradient_accumulation_steps 2 \
--overwrite_output
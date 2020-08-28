# # REQUIRED: DOWNLOAD SST FROME GLUE $ python3 transformers/utils/download_glue_data.py --tasks MNLI
# export DATA_DIR="/content/gdrive/My Drive/adapter/data/commonsenseqa"
# export TASK_NAME=commonsenseqa

# python3 run_glue_wh.py \
#   --model_name_or_path bert-base-cased \
#   --task_name $TASK_NAME \
#   --do_train \
#   --do_eval \
#   --data_dir "$DATA_DIR/$TASK_NAME" \
#   --max_seq_length 128 \
#   --per_device_train_batch_size 64 \
#   --learning_rate 1e-4 \
#   --num_train_epochs 10.0 \
#   --output_dir "/content/gdrive/My Drive/adapter/adapters/$TASK_NAME" \
#   --overwrite_output_dir \
#   --train_adapter \
#   --adapter_config pfeiffer

export DATA_DIR="/content/gdrive/My Drive/adapter/data"
export TASK_NAME=commonsenseqa

python3 run_cqa_wh.py \
--task_name $TASK_NAME \
--model_name_or_path bert-base-cased \
--do_train \
--do_eval \
--data_dir "$DATA_DIR/$TASK_NAME" \
--learning_rate 5e-5 \
--num_train_epochs 3 \
--max_seq_length 80 \
--output_dir "/content/gdrive/My Drive/adapter/adapters/$TASK_NAME" \
--per_gpu_eval_batch_size=16 \
--per_device_train_batch_size=16 \
--gradient_accumulation_steps 2 \
--overwrite_output

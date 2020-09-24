# REQUIRED: DOWNLOAD SST FROME GLUE $ python3 transformers/utils/download_glue_data.py --tasks MNLI
export GLUE_DIR="/home/theorist17/projects/adapter/data/glue_data"
export TASK_NAME=MNLI

python3 run_glue_wh.py \
  --model_name_or_path bert-base-cased \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --data_dir "$GLUE_DIR/$TASK_NAME" \
  --max_seq_length 128 \
  --per_device_train_batch_size 64 \
  --learning_rate 1e-4 \
  --num_train_epochs 10.0 \
  --output_dir "/home/theorist17/projects/adapter/adapters/$TASK_NAME" \
  --overwrite_output_dir \
  --train_adapter \
  --adapter_config pfeiffer \
  --logging_steps 1000 \
  --save_steps 1000 \
  --evaluate_during_training
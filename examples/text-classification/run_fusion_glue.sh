# REQUIRED: DOWNLOAD SST FROME GLUE $ python3 transformers/utils/download_glue_data.py --tasks MNLI
export GLUE_DIR=/path/to/glue
export TASK_NAME=SST-2

python3 run_fusion_glue.py \
  --model_name_or_path bert-base-cased \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --data_dir $GLUE_DIR/$TASK_NAME \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 5e-5 \
  --num_train_epochs 10.0 \
  --output_dir /tmp/$TASK_NAME \
  --overwrite_output_dir
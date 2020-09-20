# REQUIRED: DOWNLOAD SST FROME GLUE $ python3 transformers/utils/download_glue_data.py --tasks MNLI
cd /home/theorist17/projects/adapter/adapter-transformers
export CONCEPTNET_DIR="/home/theorist17/projects/adapter/neo4j-kbs/classification"
export TASK_NAME=conceptnet

python3 run_conceptnet.py \
  --model_name_or_path bert-base-cased \
  --do_train \
  --do_eval \
  --data_dir "$CONCEPTNET_DIR" \
  --max_seq_length 64 \
  --per_gpu_train_batch_size 128 \
  --per_gpu_eval_batch_size 128 \
  --cache_dir "/home/theorist17/projects/adapter/adapters/$TASK_NAME" \
  --num_train_epochs 10.0 \
  --evaluate_during_training \
  --do_lower_case \
  --output_dir "/home/theorist17/projects/adapter/adapters/$TASK_NAME" \
  --overwrite_output_dir \
  --num_train_epochs 3 \
  #--overwrite_cache # features!
  #--train_adapter \
  #--adapter_config pfeiffer

#  --output_dir /tmp/$TASK_NAME \
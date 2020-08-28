export TRAIN_FILE="/content/gdrive/My Drive/adapter/data/conceptnet/train100k.txt"
export TEST_FILE="/content/gdrive/My Drive/adapter/data/conceptnet/dev1.txt"

python3 run_language_modeling.py \
    --output_dir="/content/gdrive/My Drive/adapter/models/conceptnet" \
    --model_type=bert \
    --model_name_or_path=bert-base-uncased \
    --do_train \
    --train_data_file="$TRAIN_FILE"\
    --line_by_line \
    --do_eval \
    --eval_data_file="$TEST_FILE" \
    --mlm \
    --do_predict
    --overwrite_output_dir
#!/bin/sh

# change work directory to root
cd ../
pwd
F2O_BIN="./bin/pos_input1_classification_feature2output_layer"
RNN_TYPE="lstm"
ANNOTATED_FILE="example/sampledata/postag/PTB_train.50.pos"
INPUT_FILE="example/sampledata/postag/PTB_input.50"
MODEL_PATH="example/f2o.model"
PREDICT_OUTPUT="example/predict.out"
[ -e "$MODEL_PATH" ] && /bin/rm "$MODEL_PATH" # remove it if exists

# train
echo "-- training process --" >/dev/stderr
$F2O_BIN train $RNN_TYPE --cnn-mem 256 --training_data $ANNOTATED_FILE --devel_data $ANNOTATED_FILE --max_epoch 5 --dropout_rate 0 --model $MODEL_PATH

# devel
echo "-- devel process --" >/dev/stderr
$F2O_BIN devel $RNN_TYPE --cnn-mem 256 --devel_data $ANNOTATED_FILE --model $MODEL_PATH

# predict 
echo "-- predict process --" >/dev/stderr
$F2O_BIN predict $RNN_TYPE --cnn-mem 256 --raw_data $INPUT_FILE --model $MODEL_PATH --output $PREDICT_OUTPUT


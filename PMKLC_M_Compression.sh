#!/bin/bash
# ====================================================== #
# authors: DYF
# date:    2024/04/25
# describe:shell file for running PMKLC Source Code
# example: bash PMKLC_M_Compression.sh DataSets/ScPo 0 320 3 3 SPuM
# ====================================================== #
FILE=$1       # 文件名
GPU=$2        # 算法运行在哪一块GPU
batch_size=$3 # 算法使用的batch-size大小
K=$4          # 窗口大小，对应文中k
W=$5          # 步长，对应文中s
MODEL_PATH=$6 # SPuM
FILE_SIZE=$(ls -ll $FILE | awk '{print  $5}')
if [ "$FILE_SIZE" -lt 500000000 ]; then
    MODEL_LIST_NUM=5
else
    MODEL_LIST_NUM=3
fi
echo "File size: "$FILE_SIZE", use mode "$MODEL_LIST_NUM
BASE=${FILE##*/}
BASE=${BASE%.*}
OUTPUT=${BASE}_${K}_${W}.pmklc
static_public_model_path=${MODEL_PATH}"/all_model_${K}_${W}" # Static Public Model所在路径


# 无论哪一种模式，均需要运行(s,k)-mer 编码脚本
# python run_k.py --file_name $FILE --dictionary_encoding_k $K --dictionary_encoding_w $W --output_file_num 4
echo "-----------------Running GPU (s,k)-mer------------------"
./SKMER_M $FILE $K $W $GPU 4

static_private_model="SPrM/${BASE}_SPrM_${K}_${W}"
if [ "$MODEL_LIST_NUM" != 1 ] && [ "$MODEL_LIST_NUM" != 4 ] && [ "$MODEL_LIST_NUM" != 5 ]; then
    echo "-------------Training static private model--------------"
    python train_SPrM.py --k $K --w $W --file_name $BASE --timesteps 32 --model_weights_path ${static_private_model} --gpu $GPU
fi
echo "--------Compressing ${BASE} with model-list-number:${MODEL_LIST_NUM}--------"

WATCH_DIR=${BASE}"_model"

if [ -d "$WATCH_DIR" ]; then
    rm -rf "$WATCH_DIR"
fi
mkdir -p "$WATCH_DIR"

pids=""
if [ -z "$(ls -A "$WATCH_DIR")" ]; then
    python CPMKLC_M.py --save --bs ${batch_size} --model_catena ${MODEL_PATH} --k $K --w $W --file_name ${BASE} --timesteps 32 --output ${OUTPUT} --static_public_model_path ${static_public_model_path} --static_private_model_path ${static_private_model} --gpu 0 --model_list_num ${MODEL_LIST_NUM} &
    pid=$!
    pids="$pids $pid"
fi

for i in 1 2 3; do
    TARGET_FILE="${BASE}.$(expr $i - 1).pth"
    while true; do
        if [ -e "$WATCH_DIR/$TARGET_FILE" ]; then
            python CPMKLC_M.py --save --load --bs ${batch_size} --model_catena ${MODEL_PATH} --k $K --w $W --file_name ${BASE} --timesteps 32 --output ${OUTPUT} --static_public_model_path ${static_public_model_path} --static_private_model_path ${static_private_model} --gpu $i --model_list_num ${MODEL_LIST_NUM} &
            pid=$!
            pids="$pids $pid"
            break
        else
            sleep 0.1
        fi
    done
done

for pid in $pids; do
    wait $pid
done

echo "---------------Combined and merge result----------------"
tar -czf ${OUTPUT} 0_${OUTPUT}.combined 1_${OUTPUT}.combined 2_${OUTPUT}.combined 3_${OUTPUT}.combined

python merge_param.py --input_pattern "${OUTPUT}*.params" --output_file "${OUTPUT}.params"

sed -i 'N;$!P;$!D;$s/\n\(.*}\)/,\n\1/' "${OUTPUT}.params"
sed -i '/}$/i \    "model_mode_num": '"$MODEL_LIST_NUM"'' "${OUTPUT}.params"

mv ${BASE}_${K}_${W}.pmklc ${BASE}_${K}_${W}.pmklc.combined
rm ${OUTPUT}?.params
rm ${BASE}_${K}_${W}.npy
rm 0_*
rm 1_*
rm 2_*
rm 3_*
rm -rf ${BASE}_model
rm params_${BASE}*

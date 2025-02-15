#!/bin/bash
# ====================================================== #
# authors: XXX
# date:    2024/04/25
# describe:shell file for running PMKLC Source Code
# example: bash PMKLC_S_Compression.sh /home/dyf/PMKLC/DataSets/ScPo 0 320 3 3 SPuM
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
echo "-----------------Running GPU (s,k)-mer------------------"
# python SKMER_S.py --file_name $FILE --dictionary_encoding_k $K --dictionary_encoding_w $W
./SKMER_S $FILE $K $W $GPU

static_private_model="SPrM/${BASE}_SPrM_${K}_${W}"
if [ "$MODEL_LIST_NUM" != 1 ] && [ "$MODEL_LIST_NUM" != 4 ] && [ "$MODEL_LIST_NUM" != 5 ]; then
    echo "-------------Training static private model--------------"
    python train_SPrM.py --k $K --w $W --file_name $BASE --timesteps 32 --model_weights_path ${static_private_model} --gpu $GPU
fi
echo "--------Compressing ${BASE} with model-list-number:${MODEL_LIST_NUM}--------"
python CPMKLC_S.py --bs ${batch_size} --model_catena ${MODEL_PATH} --k $K --w $W --file_name ${BASE} --bs ${batch_size} --timesteps 32 --output ${OUTPUT} --static_public_model_path ${static_public_model_path} --static_private_model_path ${static_private_model} --gpu $GPU --model_list_num ${MODEL_LIST_NUM}

sed -i 'N;$!P;$!D;$s/\n\(.*}\)/,\n\1/' "${OUTPUT}.params"
sed -i '/}$/i \    "model_mode_num": '"$MODEL_LIST_NUM"'' "${OUTPUT}.params"

rm ${BASE}_${K}_${W}.npy
rm params_${BASE}*

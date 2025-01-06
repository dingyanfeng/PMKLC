# =============================================================== #
# authors: DYF
# date:    2024/04/26
# describe:a shell file for KDMSDLC algorithm decompression
# example: bash PMKLC_S_Decompression.sh ScPo_3_3.pmklc.combined 0 3 3 SPuM
# =============================================================== #
FILE=$1       # 文件名
GPU=$2        # 算法运行在哪一块GPU
K=$3          # 窗口大小，对应文中k
W=$4          # 步长，对应文中s
MODEL_PATH=$5 # 采用哪个系列模型 SPuM系列
BASE=${FILE##*/}
BASE=${BASE%.*}
MODEL_LIST_NUM=$(grep -oP '"model_mode_num":\s*\K\d+' "${BASE}.params")
OUTPUT=${BASE}_recover
static_public_model_path=${MODEL_PATH}"/all_model_${K}_${W}" # Static Public Model所在路径

echo "-------Decompressing ${BASE} with model-list-number:${MODEL_LIST_NUM}-------"

NAME=$(echo "$BASE" | cut -d'_' -f1)
static_private_model="SPrM/${NAME}_SPrM_${K}_${W}"
python DCPMKLC_S.py --model_catena ${MODEL_PATH} --file_name ${BASE} --output ${OUTPUT} --static_public_model_path ${static_public_model_path} --static_private_model_path ${static_private_model} --k ${K} --w ${W} --gpu ${GPU} --model_list_num ${MODEL_LIST_NUM}

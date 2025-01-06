# =============================================================== #
# authors: DYF
# date:    2024/04/26
# describe:a shell file for PMKLC algorithm decompression
# example: bash PMKLC_M_Decompression.sh ScPo_3_3.pmklc.combined 0 3 3 SPuM
# =============================================================== #
FILE=$1       # 文件名
GPU=$2        # 算法运行在哪一块GPU
K=$3          # 窗口大小，对应文中k
W=$4          # 步长，对应文中w
MODEL_PATH=$5 # SPuM
BASE=${FILE##*/}
BASE=${BASE%.*}
MODEL_LIST_NUM=$(grep -oP '"model_mode_num":\s*\K\d+' "${BASE}.params")
OUTPUT=${BASE}_recover                        # 根据编码方案计算的输出的压缩文件名
static_public_model_path=${MODEL_PATH}"/all_model_${K}_${W}" # Static Public Model所在的文件路径

echo "-------Decompressing ${BASE} with model-list-number:${MODEL_LIST_NUM}-------"
NAME=$(echo "$BASE" | cut -d'_' -f1)
static_private_model="SPrM/${NAME}_SPrM_${K}_${W}"

tar -xvf ${BASE}.combined

WATCH_DIR=${BASE}"_model"

if [ -d "$WATCH_DIR" ]; then
    rm -rf "$WATCH_DIR"
fi
mkdir -p "$WATCH_DIR"

pids=""
if [ -z "$(ls -A "$WATCH_DIR")" ]; then
    I_OUTPUT="0_"${OUTPUT}
    python DCPMKLC_M.py --save --model_catena ${MODEL_PATH} --file_name ${BASE} --output ${I_OUTPUT} --static_public_model_path ${static_public_model_path} --static_private_model_path ${static_private_model} --k ${K} --w ${W} --gpu 0 --model_list_num ${MODEL_LIST_NUM} &
    pid=$!
    pids="$pids $pid"
fi

for i in 1 2 3; do
    TARGET_FILE="${BASE}.$(expr $i - 1).pth"
    while true; do
        if [ -e "$WATCH_DIR/$TARGET_FILE" ]; then
            I_OUTPUT=${i}"_"${OUTPUT}
            python DCPMKLC_M.py --save --load --model_catena ${MODEL_PATH} --file_name ${BASE} --output ${I_OUTPUT} --static_public_model_path ${static_public_model_path} --static_private_model_path ${static_private_model} --k ${K} --w ${W} --gpu $i --model_list_num ${MODEL_LIST_NUM} &
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

PARAM_FILE="${BASE}.params"
WRITE_CHARS=$(grep -oP '"Write-Chars": "\K[^"]*' "$PARAM_FILE")

cat 0_${OUTPUT} 1_${OUTPUT} 2_${OUTPUT} 3_${OUTPUT} > ${OUTPUT}

echo -n ${WRITE_CHARS} >> ${OUTPUT}

rm -rf ${BASE}_model
rm 0_*
rm 1_*
rm 2_*
rm 3_*


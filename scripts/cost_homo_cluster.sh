# Copyright 2024 Samsung Electronics Co., Ltd. All Rights Reserved
#!/bin/bash

cd "$(dirname $"0")"

for ARGUMENT in "$@"
do
   KEY=$(echo $ARGUMENT | cut -f1 -d=)

   KEY_LENGTH=${#KEY}
   VALUE="${ARGUMENT:$KEY_LENGTH+1}"

   export "$KEY"="$VALUE"
done

model_options="
                --model_name=${MODEL_NAME}
                --num_layers=${NUM_LAYERS}
                --gbs=${GBS}
              "

HOST_FILE_PATH="${HOME_DIR}/hostfile"
CLUSTER_INFO_FILE_PATH="${HOME_DIR}/clusterfile.json"

cluster_options="
                  --hostfile_path=${HOST_FILE_PATH}
                  --clusterfile_path=${CLUSTER_INFO_FILE_PATH}
                "

LOG_PATH="${HOME_DIR}/logs"
current_time=$(date "+%Y.%m.%d-%H.%M.%S")

env_options="
              --home_dir=${HOME_DIR}
              --log_path=${LOG_PATH}
            "

PROFILE_DATA_PATH="${HOME_DIR}/profile"

hetspeed_options="
                    --profile_data_path=${PROFILE_DATA_PATH}
                    --max_profiled_tp_degree=${MAX_PROFILED_TP}
                    --max_profiled_batch_size=${MAX_PROFILED_BATCH_SIZE}
                 "

run_cmd="python3 ../cost_homo_cluster.py ${model_options} ${cluster_options} ${hetspeed_options} ${env_options}
         &> ${LOG_PATH}/${MODEL_NAME}_${MODEL_SIZE}_${current_time}.log"

echo ${run_cmd}
eval ${run_cmd}
set +x

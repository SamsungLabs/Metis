# Copyright 2024 Samsung Electronics Co., Ltd. All Rights Reserved
#!/bin/bash

# Change directory to the script's location
cd "$(dirname "$0")"

# Parse and export arguments
for ARGUMENT in "$@"
do
   KEY=$(echo "$ARGUMENT" | cut -f1 -d=)
   VALUE="${ARGUMENT#*=}"
   export "$KEY"="$VALUE"
done

# Define model options
model_options="\
                --model_name=${MODEL_NAME} \
                --model_size=${MODEL_SIZE} \
                --num_layers=${NUM_LAYERS} \
                --gbs=${GBS} \
              "

# Set specific options if the model is GPT
if [ "${MODEL_NAME}" == "GPT" ]; then
  if [ "${MODEL_SIZE}" == "1.5B" ]; then
    HIDDEN_SIZE=4096
    SEQUENCE_LENGTH=1024
    NUM_LAYERS=10
    VOCAB_SIZE=51200
    ATTENTION_HEAD_SIZE=32
  fi

  model_specific_options="\
                --hidden_size=${HIDDEN_SIZE} \
                --sequence_length=${SEQUENCE_LENGTH} \
                --vocab_size=${VOCAB_SIZE} \
              "
fi

# Define paths for the host and cluster files
HOST_FILE_PATH="${HOME_DIR}/hostfile"
CLUSTER_INFO_FILE_PATH="${HOME_DIR}/clusterfile.json"

cluster_options="\
                  --hostfile_path=${HOST_FILE_PATH} \
                  --clusterfile_path=${CLUSTER_INFO_FILE_PATH} \
                "

# Define log path and current timestamp
LOG_PATH="${HOME_DIR}/logs"
mkdir -p "$LOG_PATH"  # Ensure the log directory exists
current_time=$(date "+%Y.%m.%d-%H.%M.%S")

# Set environment options
env_options="\
              --home_dir=${HOME_DIR} \
              --log_path=${LOG_PATH} \
            "

# Profile data path and hetspeed options
PROFILE_DATA_PATH="${HOME_DIR}/profile"
mkdir -p "$PROFILE_DATA_PATH"  # Ensure the profile directory exists

hetspeed_options="\
                    --profile_data_path=${PROFILE_DATA_PATH} \
                    --max_profiled_tp_degree=${MAX_PROFILED_TP} \
                    --max_profiled_batch_size=${MAX_PROFILED_BATCH_SIZE} \
                    --min_group_scale_variance=${SCALE_VARIANCE} \
                    --max_permute_len=${MAX_PERMUTE_LEN} \
                 "

# Construct and run command
run_cmd="python3 ../cost_het_cluster.py ${model_options} ${model_specific_options} ${cluster_options} ${hetspeed_options} ${env_options} > ${LOG_PATH}/${MODEL_NAME}_${MODEL_SIZE}.log 2>&1"

echo "${run_cmd}"
eval "${run_cmd}"
set +x
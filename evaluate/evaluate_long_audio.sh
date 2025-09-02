#!/bin/bash

echo $ENV_VENUS_PROXY
export NO_PROXY=localhost,.woa.com,.oa.com,.tencent.com
export HTTP_PROXY=$ENV_VENUS_PROXY
export HTTPS_PROXY=$ENV_VENUS_PROXY
export no_proxy=$NO_PROXY
export http_proxy=$ENV_VENUS_PROXY
export https_proxy=$ENV_VENUS_PROXY

###############################################################################
TEST_JSON='datasets_audio_json/eval/eval_multi_audio_unav_testset_duration_dounew.json'
SAVE_FOLDER_NAME='audiostory_multi_audio_duration'
SOURCE_AUDIO='data/FAD_UnAV_testset_32000'
AUDIO_TYPE='audio_gen_long_audio_correct'

GUIDANCE=4.0
crossfade_sec=3.0


CUDA_VISIBLE_DEVICES=0 python evaluate/evaluate_long_audio.py \
                                    --audio_type ${AUDIO_TYPE} \
                                    --crossfade_sec ${crossfade_sec} \
                                    --testset_json ${TEST_JSON} \
                                    --model_path ${MODEL_PATH} \
                                    --guidance ${GUIDANCE} \
                                    --save_folder_name ${SAVE_FOLDER_NAME}
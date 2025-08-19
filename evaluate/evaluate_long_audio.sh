#!/bin/bash

echo $ENV_VENUS_PROXY
export NO_PROXY=localhost,.woa.com,.oa.com,.tencent.com
export HTTP_PROXY=$ENV_VENUS_PROXY
export HTTPS_PROXY=$ENV_VENUS_PROXY
export no_proxy=$NO_PROXY
export http_proxy=$ENV_VENUS_PROXY
export https_proxy=$ENV_VENUS_PROXY

###############################################################################
REPO_ID='/group/40034/gloriayxguo/TangoFlux_SEED/checkpoints/TangoFlux'
TEST_JSON='datasets_audio_json/eval/eval_multi_audio_unav_testset_duration_dounew.json'
SAVE_FOLDER_NAME='audiostory_multi_audio_duration'
SOURCE_AUDIO='data/FAD_UnAV_testset_32000'
AUDIO_TYPE='audio_gen_long_audio_correct'

GUIDANCE=4.0
crossfade_sec=3.0

cd /group/40034/gloriayxguo/AudioStory_open
source /data/miniconda3/bin/activate seedx
# pip install pydub diffusers==0.30.0 librosa==0.9.2

################################### generate pytorch bin ############################################ 
MODEL_PATH='audioseed_ckpt/seed_omni_t5_multi_audio_duration/audiostory_qwen_3b_t5_multi_audio_unav_scale10_1e4_loss0105_bz8_genpretrain_withinst_t5_aud_attn_cotrain_with_mhattn_weight_detokenizer_full_open_1opt_coscale_8token_duration_begin0_new/checkpoint-12000'
FILE="$MODEL_PATH/pytorch_model.bin"

# 判断文件是否存在
if [ -f "$FILE" ]; then
    echo "$FILE 文件存在"
else
    echo "$FILE 文件不存在"
    cd ${MODEL_PATH}
    CUDA_VISIBLE_DEVICES=0 python zero_to_fp32.py . pytorch_model.bin
    cd /group/40034/gloriayxguo/AudioStory_open
fi


CUDA_VISIBLE_DEVICES=0 python evaluate/evaluate_long_audio.py \
                                    --audio_type ${AUDIO_TYPE} \
                                    --crossfade_sec ${crossfade_sec} \
                                    --testset_json ${TEST_JSON} \
                                    --model_path ${MODEL_PATH} \
                                    --guidance ${GUIDANCE} \
                                    --save_folder_name ${SAVE_FOLDER_NAME}
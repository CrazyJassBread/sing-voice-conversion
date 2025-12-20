#!/bin/bash

source='/home/test/ac/codes/sing-voice-conversion/datasets/source/infer/王力宏 - 爱错/vocals.wav'
accompany='/home/test/ac/codes/sing-voice-conversion/datasets/source/infer/王力宏 - 爱错/instrumental.wav'
target="/home/test/ac/codes/sing-voice-conversion/datasets/target_hajimi/new-0.wav"
diffusion_step=100
fp16="True"
config='./configs/YingMusic-SVC.yml'
# checkpoint='/home/test/ac/codes/sing-voice-conversion/YingMusic-SVC/checkpoints/YingMusic/YingMusic-SVC-full.pt'
checkpoint='/home/test/ac/codes/sing-voice-conversion/YingMusic-SVC/runs/my_exp/ft_model.pth'
expname="sfa_lora"
cuda="0"


python my_inference_new.py \
    --source "${source}" \
    --target "${target}" \
    --diffusion-steps "${diffusion_step}" \
    --checkpoint "${checkpoint}" \
    --expname "${expname}" \
    --cuda "${cuda}" \
    --fp16 "${fp16}" \
    --accompany "${accompany}" \
    --config "${config}"\
    --use-lora 



# --use-rva 
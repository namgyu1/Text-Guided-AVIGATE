#!/bin/bash
# Resume training from step 800 checkpoint

export MKL_NUM_THREADS=24
export NUMEXPR_NUM_THREADS=24
export OMP_NUM_THREADS=24
DATA_PATH=../datasets/MSRVTT
RPort=$(shuf -i 1000-9999 -n1)
Margin=0.1
beta=0.2
CKPT_NAME=TAVIGATE_L2
Tau=1.0

# Resume from step 800 checkpoint
STEP_CKPT=1000
EPOCH_NUM=1002

echo "Resuming training from step ${STEP_CKPT} checkpoint"
echo "Loading model: ckpts/${CKPT_NAME}/pytorch_model.bin.step_${STEP_CKPT}.${EPOCH_NUM}"
echo "Loading optimizer: ckpts/${CKPT_NAME}/pytorch_opt.bin.step_${STEP_CKPT}.${EPOCH_NUM}"

OMP_NUM_THREADS=48 CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --master_port $RPort --nproc_per_node=4 main_task_retrieval.py --do_train --num_thread_reader=12 \
    --epochs=5 --batch_size=128 --n_display=50 --checkpoint_interval=200 --train_csv ${DATA_PATH}/MSRVTT_train.9k.csv --val_csv ${DATA_PATH}/MSRVTT_JSFUSION_test.csv \
    --data_path ${DATA_PATH}/MSRVTT_data.json --features_path ${DATA_PATH}/videos/all_compressed --audio_path ${DATA_PATH}/videos/audio_all_compressed --output_dir ckpts/${CKPT_NAME} --lr 5e-5 \
    --max_words 32 --max_frames 12 --batch_size_val 16 --datatype msrvtt --expand_msrvtt_sentences --feature_framerate 1 --coef_lr 1e-3 --freeze_layer_num 12  \
    --slice_framepos 2 --loose_type --linear_patch 2d --sim_header seqTransf --pretrained_clip_name ViT-B/32 --eval_max_frame 12 --temperature $Tau --warmup_proportion 0.1 --cross_num_hidden_layers 4 --audio_query_layers 4 --beta $beta --margin_BD $Margin --gradient_accumulation_steps 1 \
    --resume_model ckpts/${CKPT_NAME}/pytorch_opt.bin.step_${STEP_CKPT}.${EPOCH_NUM} --init_model ckpts/${CKPT_NAME}/pytorch_model.bin.step_${STEP_CKPT}.${EPOCH_NUM}

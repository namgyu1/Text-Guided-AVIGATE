export MKL_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export OMP_NUM_THREADS=6
DATA_PATH=../datasets/VATEX
RPort=$(shuf -i 1000-9999 -n1)
# OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --master_port 1222 --nproc_per_node=4 main_task_retrieval.py --do_train --num_thread_reader=12 \
#     --epochs=5 --batch_size=128 --n_display=10 --train_csv ${DATA_PATH}/MSRVTT_train.9k.csv --val_csv ${DATA_PATH}/MSRVTT_JSFUSION_test.csv \
#     --data_path ${DATA_PATH}/MSRVTT_data.json --features_path ${DATA_PATH}/videos/all --output_dir ckpts/ckpt_msrvtt_retrieval_set_based_LMH_s32 --lr 1e-4 \
#     --max_words 32 --max_frames 12 --batch_size_val 32 --datatype msrvtt   --feature_framerate 1 --coef_lr 1e-3 --freeze_layer_num 0  \
#     --slice_framepos 2 --loose_type --linear_patch 2d --sim_header meanP --pretrained_clip_name ViT-B/32 --eval_max_frame 12
    
# ONP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --master_port 1221 --nproc_per_node=4 main_task_retrieval.py --do_train --num_thread_reader=1 \
#     --epochs=5 --batch_size=128 --n_display=50 --train_csv ${DATA_PATH}/MSRVTT_train.9k.csv --val_csv ${DATA_PATH}/MSRVTT_JSFUSION_test.csv \
#     --data_path ${DATA_PATH}/MSRVTT_data.json --features_path ${DATA_PATH}/videos/all --output_dir ckpts/ckpt_msrvtt_retrieval_AdaptFormer_8 --lr 1e-5 \
#     --max_words 32 --max_frames 12 --batch_size_val 32 --datatype msrvtt --expand_msrvtt_sentences  --feature_framerate 1 --coef_lr 1 --freeze_layer_num 0  \
#     --slice_framepos 2 --loose_type --linear_patch 3d --sim_header meanP --pretrained_clip_name ViT-B/32 --eval_max_frame 12


# ONP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=4 python -m torch.distributed.launch --master_port 1222 --nproc_per_node=1 main_task_retrieval.py --do_train --num_thread_reader=12 \
#     --epochs=5 --batch_size=32 --n_display=50 --train_csv ${DATA_PATH}/MSRVTT_train.9k.csv --val_csv ${DATA_PATH}/MSRVTT_JSFUSION_test.csv \
#     --data_path ${DATA_PATH}/MSRVTT_data.json --features_path ${DATA_PATH}/videos/all --output_dir ckpts/ckpt_msrvtt_retrieval_BOCO_Pad3_Top02_Adapter --lr 1e-5 \
#     --max_words 32 --max_frames 12 --batch_size_val 32 --datatype msrvtt --expand_msrvtt_sentences  --feature_framerate 1 --coef_lr 1 --freeze_layer_num 0  \
#     --slice_framepos 2 --loose_type --linear_patch 2d --sim_header meanP --pretrained_clip_name ViT-B/32 --eval_max_frame 12

# OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=4 python -m torch.distributed.launch --master_port 1222 --nproc_per_node=1 main_task_retrieval.py --do_eval --num_thread_reader=12 \
#     --epochs=5 --batch_size=128 --n_display=50 --train_csv ${DATA_PATH}/MSRVTT_train.9k.csv --val_csv ${DATA_PATH}/MSRVTT_JSFUSION_test.csv \
#     --data_path ${DATA_PATH}/MSRVTT_data.json --features_path ${DATA_PATH}/videos/all --output_dir ckpts/ckpt_msrvtt_retrieval_set_based --lr 1e-4 \
#     --max_words 32 --max_frames 12 --batch_size_val 32 --datatype msrvtt   --feature_framerate 1 --coef_lr 1e-3 --freeze_layer_num 0  \
#     --slice_framepos 2 --loose_type --linear_patch 2d --sim_header meanP --pretrained_clip_name ViT-B/32 --eval_max_frame 12 --init_model ckpts/ckpt_msrvtt_retrieval_set_based/pytorch_model.bin.0

# ckpt_msrvtt_retrieval_Analysis_VT_Neg_0821
 
####################################################################################################################################################################################################
# Margin=0.05
# beta=0.2
 
# CKPT_NAME=ckpt_vatex_AQ_4_Transf_4_12tokens_BiMaxThsMarginAll0.05_Beta0.2_residualFuse95_CLIPVisual_gain05_woLN_CrossFirst_Aux01_TVA_FusedGLobal_1110
# Tau=1.0
# OMP_NUM_THREADS=48 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --master_port 1222 --nproc_per_node=8 main_task_retrieval.py --do_train --num_thread_reader=12 \
#     --epochs=5 --batch_size=128 --n_display=50 --data_path ${DATA_PATH} --features_path ${DATA_PATH}/videos_all_compressed --audio_path ${DATA_PATH}/audio_all_compressed --output_dir ckpts/${CKPT_NAME} --lr 1e-4 \
#     --max_words 32 --max_frames 12 --batch_size_val 32 --datatype vatex --feature_framerate 1 --coef_lr 1e-3 --freeze_layer_num 12  \
#     --slice_framepos 2 --loose_type --linear_patch 2d --sim_header seqTransf --pretrained_clip_name ViT-B/32 --eval_max_frame 12 --temperature $Tau --warmup_proportion 0.1 --cross_num_hidden_layers 4 --audio_query_layers 4 --beta $beta --margin_BD $Margin #--resume_model ckpts/${CKPT_NAME}/pytorch_opt.bin.1 --init_model ckpts/${CKPT_NAME}/pytorch_model.bin.1

# chmod -R 777 ckpts/*

Margin=0.05
beta=0.2
# CKPT_NAME=ckpt_vatex_AQ_4_Transf_4_12tokens_BiMaxThsMarginAll0.05_Beta0.2_residualFuse95_CLIPVisual_Aux01_TVA_Gate_1025
CKPT_NAME=ckpt_vatex_AQ_4_Transf_4_12tokens_BiMaxThsMarginAll0.05_Beta0.2_residualFuse95_CLIPVisual_Aux01_TVA_Gate_gain05_ViTB16_1108
Tau=1.0
epoch=$(seq 4 4)
for ep in $epoch
do
    OMP_NUM_THREADS=6 CUDA_VISIBLE_DEVICES=9 python -m torch.distributed.launch --master_port $RPort --nproc_per_node=1 main_task_retrieval.py --do_eval --num_thread_reader=48 \
    --epochs=5 --batch_size=128 --n_display=50 --data_path ${DATA_PATH} --features_path ${DATA_PATH}/videos_all_compressed --audio_path ${DATA_PATH}/audio_all_compressed --output_dir ckpts/${CKPT_NAME} --lr 1e-4 \
    --max_words 32 --max_frames 12 --batch_size_val 100 --datatype vatex --feature_framerate 1 --coef_lr 1e-3 --freeze_layer_num 12  \
    --slice_framepos 2 --loose_type --linear_patch 2d --sim_header seqTransf --pretrained_clip_name ViT-B/16 --eval_max_frame 12 --temperature $Tau --warmup_proportion 0.1 --cross_num_hidden_layers 4 --audio_query_layers 4 --beta $beta --margin_BD $Margin --init_model ckpts/${CKPT_NAME}/pytorch_model.bin.$ep
done
chmod -R 777 ckpts/*

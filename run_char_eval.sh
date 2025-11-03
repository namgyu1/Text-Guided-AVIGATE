export MKL_NUM_THREADS=24
export NUMEXPR_NUM_THREADS=24
export OMP_NUM_THREADS=24
DATA_PATH=../datasets/Charades
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
# CKPT_NAME=ckpt_msrvtt_Transf_4_16tokens_BiSoftLabel_Alpha1_BiSumThsMarginAll1_Beta3_residualFuse95_CLIPVisual_sumto1_woSoftAlpha_diffTemp005001_1010
# CKPT_NAME=ckpt_msrvtt_Transf_4_16tokens_Gate_AdaptiveMargin_Beta03_1016
# Tau=1.0
# OMP_NUM_THREADS=24 CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --master_port $RPort --nproc_per_node=4 main_task_retrieval.py --do_train --num_thread_reader=6 \
#     --epochs=5 --batch_size=64 --n_display=100 --train_csv ${DATA_PATH}/MSRVTT_train.9k.csv --val_csv ${DATA_PATH}/MSRVTT_JSFUSION_test.csv \
#     --data_path ${DATA_PATH}/MSRVTT_data.json --features_path ${DATA_PATH}/videos/all_compressed --audio_path ${DATA_PATH}/videos/audio_all_compressed --output_dir ckpts/${CKPT_NAME} --lr 1e-4 \
#     --max_words 32 --max_frames 12 --batch_size_val 16 --datatype msrvtt --expand_msrvtt_sentences --feature_framerate 1 --coef_lr 1e-3 --freeze_layer_num 12  \
#     --slice_framepos 2 --loose_type --linear_patch 2d --sim_header seqTransf --pretrained_clip_name ViT-B/32 --eval_max_frame 12 --temperature $Tau --warmup_proportion 0.1 --cross_num_hidden_layers 1 --audio_query_layers 4 #--resume_model ckpts/${CKPT_NAME}/pytorch_opt.bin.0 --init_model ckpts/${CKPT_NAME}/pytorch_model.bin.0

# chmod -R 777 ckpts/*

# Margin=0.1
# beta=0.2
 
# CKPT_NAME=Rebuttal_ckpt_charades_AQ_4_Transf_4_12tokens_BiMaxThsMarginAll0.1_Beta0.2_residualFuse95_CLIPVisual_gain05_woLN_CrossFirst_Aux01_TVA_0126
# Tau=1.0
# OMP_NUM_THREADS=48 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --master_port $RPort --nproc_per_node=8 main_task_retrieval.py --do_train --num_thread_reader=12 \
#     --epochs=5 --batch_size=64 --n_display=50 --train_csv ${DATA_PATH}/Charades_v1_train.csv --val_csv ${DATA_PATH}/Charades_v1_test.csv \
#     --features_path ${DATA_PATH}/Charades_v1_480 --audio_path ${DATA_PATH}/audio_all_compressed --output_dir ckpts/${CKPT_NAME} --lr 1e-5 \
#     --max_words 64 --max_frames 32 --batch_size_val 32 --datatype charades --feature_framerate 1 --coef_lr 1e-2 --freeze_layer_num 12  \
#     --slice_framepos 2 --loose_type --linear_patch 2d --sim_header seqTransf --pretrained_clip_name ViT-B/32 --eval_max_frame 32 --temperature $Tau --warmup_proportion 0.1 --cross_num_hidden_layers 4 --audio_query_layers 4 --beta $beta --margin_BD $Margin #--resume_model ckpts/${CKPT_NAME}/pytorch_opt.bin.0 --init_model ckpts/${CKPT_NAME}/pytorch_model.bin.0

Margin=0.1
beta=0.2
 
CKPT_NAME=Best_charades_Rebuttal_ckpt_charades_AQ_4_Transf_4_8tokens_BiMaxThsMarginAll0.1_Beta0.2_residualFuse95_CLIPVisual_gain05_woLN_CrossFirst_Aux01_TVA_5e4_2e4_0128
Tau=1.0
epoch=$(seq 4 4)
for ep in $epoch
do
OMP_NUM_THREADS=48 CUDA_VISIBLE_DEVICES=9 python -m torch.distributed.launch --master_port $RPort --nproc_per_node=1 main_task_retrieval.py --do_eval --num_thread_reader=12 \
    --epochs=5 --batch_size=128 --n_display=50 --train_csv ${DATA_PATH}/Charades_v1_train.csv --val_csv ${DATA_PATH}/Charades_v1_test.csv \
    --features_path ${DATA_PATH}/Charades_v1_480 --audio_path ${DATA_PATH}/audio_all_compressed --output_dir ckpts/${CKPT_NAME} --lr 1e-4 \
    --max_words 64 --max_frames 32 --batch_size_val 32 --datatype charades --feature_framerate 1 --coef_lr 1e-3 --freeze_layer_num 12  \
    --slice_framepos 2 --loose_type --linear_patch 2d --sim_header seqTransf --pretrained_clip_name ViT-B/32 --eval_max_frame 32 --temperature $Tau --warmup_proportion 0.1 --cross_num_hidden_layers 4 --audio_query_layers 4 --beta $beta --margin_BD $Margin \
    --init_model ckpts/${CKPT_NAME}/pytorch_model.bin.$ep
done
chmod -R 777 ckpts/*

# CKPT_NAME=ckpt_msrvtt_retrieval_audio_GL_AVMHSA_1018
# Tau=0.5
# # OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=4,5,6,7,8 python -m torch.distributed.launch --master_port 1222 --nproc_per_node=5 main_task_retrieval.py --do_train --num_thread_reader=4 \
# #     --epochs=5 --batch_size=40 --n_display=50 --train_csv ${DATA_PATH}/MSRVTT_train.9k.csv --val_csv ${DATA_PATH}/MSRVTT_JSFUSION_test.csv \
# #     --data_path ${DATA_PATH}/MSRVTT_data.json --features_path ${DATA_PATH}/videos/all --audio_path ${DATA_PATH}/videos/audios_all --output_dir ckpts/${CKPT_NAME} --lr 1e-4 \
# #     --max_words 32 --max_frames 12 --batch_size_val 40 --datatype msrvtt --expand_msrvtt_sentences --feature_framerate 1 --coef_lr 1e-3 --freeze_layer_num 0  \
# #     --slice_framepos 2 --loose_type --linear_patch 2d --sim_header seqTransf --pretrained_clip_name ViT-B/32 --eval_max_frame 12 --temperature $Tau --warmup_proportion 0.1 #--resume_model ckpts/${CKPT_NAME}/pytorch_opt.bin.0 --init_model ckpts/${CKPT_NAME}/pytorch_model.bin.0

# epoch=$(seq 0 4)
# for ep in $epoch
# do
#     OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --master_port 1220 --nproc_per_node=1 main_task_retrieval.py --do_eval --num_thread_reader=32 \
#         --epochs=5 --batch_size=40 --n_display=50 --train_csv ${DATA_PATH}/MSRVTT_train.9k.csv --val_csv ${DATA_PATH}/MSRVTT_JSFUSION_test.csv \
#         --data_path ${DATA_PATH}/MSRVTT_data.json --features_path ${DATA_PATH}/videos/all --audio_path ${DATA_PATH}/videos/audios_all --output_dir ckpts/${CKPT_NAME} --lr 1e-4 \
#         --max_words 32 --max_frames 12 --batch_size_val 32  --datatype msrvtt  --expand_msrvtt_sentences  --feature_framerate 1 --coef_lr 1e-3 --freeze_layer_num 0  \
#         --slice_framepos 2 --loose_type --linear_patch 2d --sim_header seqTransf --pretrained_clip_name ViT-B/32 --eval_max_frame 12 --temperature $Tau  --cross_num_hidden_layers 4 --audio_query_layers 4 --beta $beta --margin_BD $Margin --init_model ckpts/${CKPT_NAME}/pytorch_model.bin.$ep
# done
# chmod -R 777 ckpts/*

####################################################################################################################################################################################################
# CKPT_NAME=ckpt_msrvtt_retrieval_set_based_LMH_second_order_s50_independent_warmup_ep10
# epoch=$(seq 0 9)
# for ep in $epoch
# do
#     OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=9 python -m torch.distributed.launch --master_port 11222 --nproc_per_node=1 main_task_retrieval.py --do_eval --num_thread_reader=12 \
#         --epochs=5 --batch_size=128 --n_display=50 --train_csv ${DATA_PATH}/MSRVTT_train.9k.csv --val_csv ${DATA_PATH}/MSRVTT_JSFUSION_test.csv \
#         --data_path ${DATA_PATH}/MSRVTT_data.json --features_path ${DATA_PATH}/videos/all --output_dir ckpts/${CKPT_NAME} --lr 1e-4 \
#         --max_words 32 --max_frames 12 --batch_size_val 32 --datatype msrvtt   --feature_framerate 1 --coef_lr 1e-3 --freeze_layer_num 0  \
#         --slice_framepos 2 --loose_type --linear_patch 2d --sim_header meanP --pretrained_clip_name ViT-B/32 --eval_max_frame 12 --init_model ckpts/${CKPT_NAME}/pytorch_model.bin.$ep --temperature 50
# done


####################################################################################################################################################################################################
# DATA_PATH=../datasets/MSVD
# CKPT_NAME=ckpt_MSVD_retrieval_VT_attention_temp10and10_0801
# Tau=0.5
# OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --master_port 1222 --nproc_per_node=4 main_task_retrieval.py --do_train --num_thread_reader=12 \
#     --epochs=5 --batch_size=128 --n_display=50 \
#     --data_path ${DATA_PATH} --features_path ${DATA_PATH}/YouTubeClips --output_dir ckpts/${CKPT_NAME} --lr 1e-4 \
#     --max_words 32 --max_frames 12 --batch_size_val 40 --datatype msvd --feature_framerate 1 --coef_lr 1e-3 --freeze_layer_num 0  \
#     --slice_framepos 2 --loose_type --linear_patch 2d --sim_header meanP --pretrained_clip_name ViT-B/32 --eval_max_frame 12 --temperature $Tau --warmup_proportion 0.1 #--resume_model ckpts/${CKPT_NAME}/pytorch_opt.bin.2 --init_model ckpts/${CKPT_NAME}/pytorch_model.bin.2


####################################################################################################################################################################################################



# export MKL_NUM_THREADS=4
# export NUMEXPR_NUM_THREADS=4
# export OMP_NUM_THREADS=4
# export DATA_PATH=../../datasets/MSRVTT
# export CKPT_NAME=ckpt_msrvtt_retrieval_AudioQuery_4_Block_AudioVideoTransf_4_Block_TV_TAV_Real_POS_100QueryLR_AudioCELoss_AuxLoss_16tokens_agg_token_0507
# export Tau=0.5
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --master_port 1222 --nproc_per_node=4 main_task_retrieval.py --do_train --num_thread_reader=3 \
#     --epochs=5 --batch_size=128 --n_display=50 --train_csv ${DATA_PATH}/MSRVTT_train.9k.csv --val_csv ${DATA_PATH}/MSRVTT_JSFUSION_test.csv \
#     --data_path ${DATA_PATH}/MSRVTT_data.json --features_path ${DATA_PATH}/videos/all_compressed --audio_path ${DATA_PATH}/videos/audio_all_compressed --output_dir ckpts/${CKPT_NAME} --lr 5e-5 \
#     --max_words 32 --max_frames 12 --batch_size_val 32 --datatype msrvtt --expand_msrvtt_sentences --feature_framerate 1 --coef_lr 1e-3 --freeze_layer_num 12  \
#     --slice_framepos 2 --loose_type --linear_patch 2d --sim_header seqTransf --pretrained_clip_name ViT-B/32 --eval_max_frame 12 --temperature $Tau --warmup_proportion 0.1 --cross_num_hidden_layers 4


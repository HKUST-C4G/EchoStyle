# PYTHONPATH=. python -m torch.distributed.run --nproc_per_node=4 trainer/models/wan22/eval_2.25b.py \
#  --task ti2v-5B --size 1280*704  --dit_fsdp --t5_fsdp --ulysses_size 4 --ring_size 1 \
#  --ckpt_dir ../models/Wan2.2-TI2V/ \
#  --data_path ../datasets/cartoon20k/720p_1/test_sample/test_data.json \
#  --output_dir results/wan2.2_10000_tv_cfg \
#  --lora_path training_outputs/echostyle_wan2.2_5b_lora_64_fused480p/checkpoint-10000/model_state.pth

# PYTHONPATH=. python -m torch.distributed.run --nproc_per_node=4 trainer/models/wan22/eval_2.25b.py \
#  --task ti2v-5B --size 1280*704 --ulysses_size 4 --ring_size 1 \
#  --ckpt_dir ../models/Wan2.2-TI2V/ \
#  --data_path ../datasets/cartoon20k/720p_1/test_sample/test_data.json \
#  --output_dir results/wan2.2_5b_10000_61f \
#  --lora_path training_outputs/echostyle_wan2.25b_64lora_1226/checkpoint-10000/model_state.pth \
#  --frame_num 61 \
#  --sample_guide_scale 6.0 --concat True

PYTHONPATH=. python -m torch.distributed.run --nproc_per_node=8 trainer/models/wan22/eval.py \
 --task i2v-A14B --size 1280*720 --dit_fsdp --t5_fsdp --ulysses_size 8 --ring_size 1 \
 --ckpt_dir ../models/Wan2.2_I2V_14B/ \
 --data_path ../datasets/Anime20k/test.json \
 --output_dir results/final_test_10000_40step720p \
 --high_lora_path training_outputs/echostyle_v2.2_multograin_extend_high0205/checkpoint-10000/model_state.pth \
 --low_lora_path training_outputs/echostyle_v2.2_multograin_extend_low/checkpoint-20000/model_state.pth \
 --sample_guide_scale 5.0 --concat True

 PYTHONPATH=. python -m torch.distributed.run --nproc_per_node=8 trainer/models/wan22/eval.py \
 --task i2v-A14B --size 1280*720 --dit_fsdp --t5_fsdp --ulysses_size 8 --ring_size 1 \
 --ckpt_dir ../models/wan2.2_i2v_light/ \
 --data_path ../datasets/Anime20k/test.json \
 --output_dir results/final_test_10000_light20_step720p \
 --high_lora_path training_outputs/echostyle_v2.2_multograin_extend_high0205/checkpoint-10000/model_state.pth \
 --low_lora_path training_outputs/echostyle_v2.2_multograin_extend_low/checkpoint-20000/model_state.pth \
 --sample_guide_scale 1.0 --concat True --sample_steps 20

# PYTHONPATH=. python trainer/models/wan22/eval.py \
#  --task i2v-A14B --size 1280*720 \
#  --ckpt_dir ../models/Wan2.2_I2V_14B/ \
#  --data_path ../datasets/cartoon20k/720p_1/test_sample/test_data.json \
#  --output_dir results/wan-i2v-14b-multi-4000 \
#  --high_lora_path training_outputs/wan2.2_i2v_high_noise_multitask/checkpoint-4000/model_state.pth \
#  --low_lora_path training_outputs/wan2.2_i2v_low_noise_multitask/checkpoint-4000/model_state.pth \
#  --sample_guide_scale 6.0 --concat True
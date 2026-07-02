# PYTHONPATH=. python -m torch.distributed.run --nproc_per_node=8 trainer/models/wan22/eval_long.py \
#  --task i2v-A14B --size 1280*720 --dit_fsdp --t5_fsdp --ulysses_size 8 --ring_size 1 \
#  --ckpt_dir ../models/Wan2.2_I2V_14B/ \
#  --data_path ../datasets/test_movie/long_shot/test5/prompts.json \
#  --output_dir ../datasets/test_movie/long_shot/test5_vio \
#  --high_lora_path training_outputs/wan2.2_i2v_high_noise_multitask/checkpoint-16000/model_state.pth \
#  --low_lora_path training_outputs/wan2.2_i2v_low_noise_multitask/checkpoint-16000/model_state.pth \
#  --sample_guide_scale 6.0 --concat True --base_seed 42

PYTHONPATH=. python -m torch.distributed.run --nproc_per_node=8 trainer/models/wan22/eval_long.py \
 --task i2v-A14B --size 1280*720 --dit_fsdp --t5_fsdp --ulysses_size 8 --ring_size 1 \
 --ckpt_dir ../models/Wan2.2_I2V_14B/ \
 --data_path scripts/long_video/long2.json \
 --output_dir results/long/long2—caption-16000-lookback \
 --sample_guide_scale 5.0 --concat True \
 --high_extend_path training_outputs/wan2.2_i2v_high_noise_extend_0116/checkpoint-16000/model_state.pth \
 --low_extend_path training_outputs/wan2.2_i2v_low_noise_extend_0116/checkpoint-16000/model_state.pth \
 --instruct "将这段视频转化为《蓦然回首》风格。" --sample_steps 4

PYTHONPATH=. python -m torch.distributed.run --nproc_per_node=8 trainer/models/wan22/eval_long.py \
 --task i2v-A14B --size 1280*720 --dit_fsdp --t5_fsdp --ulysses_size 8 --ring_size 1 \
 --ckpt_dir ../models/Wan2.2_I2V_14B/ \
 --data_path scripts/long_video/long2.json \
 --output_dir results/long/long2—caption-16000-ghibli \
 --sample_guide_scale 5.0 --concat True \
 --high_extend_path training_outputs/wan2.2_i2v_high_noise_extend_0116/checkpoint-16000/model_state.pth \
 --low_extend_path training_outputs/wan2.2_i2v_low_noise_extend_0116/checkpoint-16000/model_state.pth \
 --instruct "将这段视频转化为《魔女宅急便》风格。" --sample_steps 4

PYTHONPATH=. python -m torch.distributed.run --nproc_per_node=8 trainer/models/wan22/eval_long.py \
 --task i2v-A14B --size 1280*720 --dit_fsdp --t5_fsdp --ulysses_size 8 --ring_size 1 \
 --ckpt_dir ../models/Wan2.2_I2V_14B/ \
 --data_path scripts/long_video/long2.json \
 --output_dir results/long/long2—caption-16000-ice \
 --sample_guide_scale 5.0 --concat True \
 --high_extend_path training_outputs/wan2.2_i2v_high_noise_extend_0116/checkpoint-16000/model_state.pth \
 --low_extend_path training_outputs/wan2.2_i2v_low_noise_extend_0116/checkpoint-16000/model_state.pth \
 --instruct "将这段视频转化为《冰雪奇缘》风格。" --sample_steps 4

PYTHONPATH=. python -m torch.distributed.run --nproc_per_node=8 trainer/models/wan22/eval_long.py \
 --task i2v-A14B --size 1280*720 --dit_fsdp --t5_fsdp --ulysses_size 8 --ring_size 1 \
 --ckpt_dir ../models/Wan2.2_I2V_14B/ \
 --data_path scripts/long_video/long2.json \
 --output_dir results/long/long2—caption-16000-violet \
 --sample_guide_scale 5.0 --concat True \
 --high_extend_path training_outputs/wan2.2_i2v_high_noise_extend_0116/checkpoint-16000/model_state.pth \
 --low_extend_path training_outputs/wan2.2_i2v_low_noise_extend_0116/checkpoint-16000/model_state.pth \
 --instruct "将这段视频转化为《薇尔莉特永恒花园》风格。" --sample_steps 4

 
import os
import imageio
import numpy as np
from tqdm import tqdm
import argparse
import json  # <-- 新增
import time  # <-- 新增
from functools import wraps  # <-- 新增
from http import HTTPStatus  # <-- 新增
import dashscope  # <-- 新增

# ==============================================================================
#  用户提供的 Prompt 生成函数 (直接复制过来)
# ==============================================================================

def retry(max_retries=3, wait_secs=1, exceptions=(Exception,)):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            retries = 0
            while True:
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    retries += 1
                    if retries > max_retries:
                        print(f"Function {func.__name__} failed after {max_retries} retries.")
                        raise
                    print(f"Function {func.__name__} failed with exception {e}. Retrying ({retries}/{max_retries})...")
                    time.sleep(wait_secs)
        return wrapper
    return decorator

@retry(max_retries=5)
def call_qwen_vl(file_path, prompt):
    is_video = file_path.endswith(".mp4")
    is_url = file_path.startswith("http")
    key = "video" if is_video else "image"
    value = file_path if is_url else f"file://{os.path.abspath(file_path)}"

    messages = [{"role": "user", "content": [{key: value}, {"text": prompt}]}]
    
    # 增加超时设置，防止请求卡死
    response = dashscope.MultiModalConversation.call(model="qwen-vl-max", messages=messages, timeout=120)

    if response.status_code == HTTPStatus.OK:
        return response.output.choices[0].message.content[0]["text"]
    else:
        raise Exception(f"Error calling Qwen-VL: {response.code} - {response.message}")

def get_prompt(file_path, qwen_vl_prompt=None, verbose=True):
    """
    从文本文件或通过 Qwen-VL 获取 prompt。
    """
    if qwen_vl_prompt is None:
        # 这个逻辑分支我们在这个脚本中不使用，但保留以保持函数完整性
        txt_path = os.path.splitext(file_path)[0] + ".txt"
        if os.path.exists(txt_path):
            with open(txt_path, "r", encoding="utf-8") as f:
                return f.read().strip()
        else:
            filename = os.path.basename(file_path)
            print(f"Warning: {txt_path} not found, skipping {filename}")
            return None
    else:
        # 核心逻辑：调用 Qwen-VL
        try:
            prompt = call_qwen_vl(file_path, qwen_vl_prompt)
            if verbose:
                print(f"  - Qwen-VL prompt for '{os.path.basename(file_path)}': {prompt}")
            return prompt
        except Exception as e:
            print(f"  - Failed to generate prompt for '{os.path.basename(file_path)}'. Error: {e}")
            return None


# ==============================================================================
#  核心的视频切分函数 (修改后)
# ==============================================================================

def split_long_video(
    video_path: str,
    output_dir: str,
    target_fps: int,
    window_size: int,
    overlap: int,
    min_tail_length: int,
    output_format: str,
    generate_prompts: bool, # <-- 新增
    qwen_prompt: str       # <-- 新增
):
    """
    将长视频重采样、切分，并选择性地为每个块生成 prompt。
    """
    # (这部分代码与之前版本相同，为了简洁省略)
    # ...
    # --- 1. 准备工作 ---
    if not os.path.exists(video_path):
        print(f"错误：视频文件不存在于 '{video_path}'")
        return
    os.makedirs(output_dir, exist_ok=True)
    print(f"输出文件夹: '{output_dir}'")
    try:
        reader = imageio.get_reader(video_path)
        original_fps = reader.get_meta_data().get('fps', 30)
        total_original_frames = reader.count_frames()
    except Exception as e:
        print(f"错误：无法读取视频文件 '{video_path}'。文件可能已损坏。错误信息: {e}")
        return
    print(f"原始视频 -> 总帧数: {total_original_frames}, 帧率 (FPS): {original_fps:.2f}")
    print(f"目标帧率: {target_fps} FPS")
    # --- 2. 计算需要挑选的帧的索引 (实现重采样) ---
    frame_step = original_fps / target_fps
    resampled_frame_indices = []
    current_frame_pos = 0.0
    while int(current_frame_pos) < total_original_frames:
        resampled_frame_indices.append(int(current_frame_pos))
        current_frame_pos += frame_step
    total_resampled_frames = len(resampled_frame_indices)
    print(f"重采样后 -> 虚拟总帧数: {total_resampled_frames}")
    # --- 3. 在重采样后的虚拟视频上计算切片索引 ---
    if total_resampled_frames < min_tail_length:
        print("重采样后视频总长度小于最小要求长度，不进行切分。")
        reader.close()
        return
    stride = window_size - overlap
    chunk_definitions = []
    start_index_in_resampled = 0
    while start_index_in_resampled + window_size <= total_resampled_frames:
        end_index_in_resampled = start_index_in_resampled + window_size
        chunk_definitions.append((start_index_in_resampled, end_index_in_resampled))
        start_index_in_resampled += stride
    remaining_len = total_resampled_frames - start_index_in_resampled
    if remaining_len >= min_tail_length:
        chunk_definitions.append((start_index_in_resampled, total_resampled_frames))
    if not chunk_definitions:
        print("根据设置，未能切分出任何视频块。")
        reader.close()
        return
    print(f"计划切分 {len(chunk_definitions)} 个视频块...")

    # --- 4. 遍历、保存视频，并生成 prompt ---
    json_data = [] # <-- 新增: 用于存储所有 prompt 数据
    
    progress_bar = tqdm(chunk_definitions, desc="正在处理视频块")
    for i, (start_resampled, end_resampled) in enumerate(progress_bar):
        # 构造输出路径
        output_filename = f"chunk_{i:03d}{output_format}"
        output_path = os.path.join(output_dir, output_filename)
        
        # 更新进度条描述
        progress_bar.set_description(f"保存 {output_filename}")

        # 保存视频块
        writer = imageio.get_writer(output_path, fps=target_fps)
        original_indices_for_chunk = resampled_frame_indices[start_resampled:end_resampled]
        for original_frame_index in original_indices_for_chunk:
            frame = reader.get_data(original_frame_index)
            writer.append_data(frame)
        writer.close()

        # --- 新增: 调用 Qwen-VL 生成 Prompt ---
        if generate_prompts:
            progress_bar.set_description(f"生成Prompt for {output_filename}")
            
            generated_prompt = get_prompt(output_path, qwen_vl_prompt=qwen_prompt)
            
            if generated_prompt:
                # 将路径和 prompt 添加到我们的列表中
                json_data.append({
                    "src_video_path": output_path,
                    "prompt": generated_prompt
                })

    reader.close()
    
    # --- 5. 最后，将收集到的数据写入 JSON 文件 ---
    if generate_prompts and json_data:
        json_output_path = os.path.join(output_dir, "prompts.json")
        print(f"\n正在将 {len(json_data)} 条 prompts 写入到 {json_output_path}...")
        with open(json_output_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=4)
        print("写入完成！")
        
    print("\n所有任务已成功完成！")


def main():
    parser = argparse.ArgumentParser(description="将长视频重采样、切分，并可选地为每个块生成prompt。")
    # ... 已有参数 ...
    parser.add_argument("--video_path", type=str, required=True, help="输入长视频文件路径。")
    parser.add_argument("--output_dir", type=str, required=True, help="保存短视频的文件夹路径。")
    parser.add_argument("--target_fps", type=int, default=16, help="重采样后的目标帧率。")
    parser.add_argument("--window_size", type=int, default=81, help="每个视频块的帧数。")
    parser.add_argument("--overlap", type=int, default=16, help="相邻块重叠的帧数。")
    parser.add_argument("--min_tail_length", type=int, default=5, help="最短尾部块长度。")
    parser.add_argument("--output_format", type=str, default=".mp4", help="输出视频格式。")
    
    # --- 新增命令行参数 ---
    parser.add_argument(
        "--generate_prompts",
        action="store_true", # 当出现这个参数时，其值为 True
        help="为每个切分后的视频块调用Qwen-VL生成prompt。"
    )
    parser.add_argument(
        "--qwen_prompt",
        type=str,
        default="用中文详细描述这个视频的内容,并避免描述风格。",
        help="向Qwen-VL提问的指令性prompt。"
    )
    
    args = parser.parse_args()

    # 检查是否需要生成 prompt，并提醒用户设置 API Key
    if args.generate_prompts and 'DASHSCOPE_API_KEY' not in os.environ:
        print("\n警告：检测到 --generate_prompts 参数，但未找到'DASHSCOPE_API_KEY'环境变量。")
        print("请先设置API密钥，例如: export DASHSCOPE_API_KEY='sk-your-key'\n")
        return

    # 调用核心函数
    split_long_video(
        video_path=args.video_path,
        output_dir=args.output_dir,
        target_fps=args.target_fps,
        window_size=args.window_size,
        overlap=args.overlap,
        min_tail_length=args.min_tail_length,
        output_format=args.output_format,
        generate_prompts=args.generate_prompts, # <-- 传递新参数
        qwen_prompt="请描述这个视频的内容，并避免描述风格"           # <-- 传递新参数
    )

if __name__ == "__main__":
    main()

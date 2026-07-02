import os
import re
import imageio
from tqdm import tqdm
import argparse

def stitch_videos_by_number(
    input_dir: str,
    output_filename: str = "stitched_video.mp4",
    overlap: int = 4,
    exclude_suffix: str = "_concat.mp4"
):
    """
    在文件夹中查找视频块，按数字排序，并以“平分重叠”的方式拼接。

    Args:
        input_dir (str): 包含视频块的文件夹路径。
        output_filename (str, optional): 输出的完整视频的文件名。
        overlap (int, optional): 相邻视频块之间重叠的帧数。
        exclude_suffix (str, optional): 处理时要排除的文件后缀。
    """
    # --- 0. 检查参数 ---
    if overlap % 2 != 0:
        print(f"警告：重叠帧数 'overlap' ({overlap}) 是奇数。建议使用偶数以实现精确平分。")
        print("将向下取整进行分割。")
    
    overlap_half = overlap // 2

    # --- 1. 查找、解析并准备排序 ---
    if not os.path.isdir(input_dir):
        print(f"错误：输入目录 '{input_dir}' 不存在或不是一个文件夹。")
        return

    number_pattern = re.compile(r'(\d+)')
    files_to_stitch = []
    for filename in os.listdir(input_dir):
        if filename.endswith(exclude_suffix) or not filename.lower().endswith('.mp4'):
            continue
        match = number_pattern.search(filename)
        if match:
            chunk_num = int(match.group(1))
            full_path = os.path.join(input_dir, filename)
            files_to_stitch.append((chunk_num, full_path))
        else:
            print(f"警告：无法从文件名 '{filename}' 中提取数字，将跳过此文件。")

    if not files_to_stitch:
        print(f"在目录 '{input_dir}' 中没有找到包含数字的 .mp4 文件。")
        return

    # --- 2. 严格按数字排序 ---
    files_to_stitch.sort(key=lambda x: x[0])
    
    print(f"找到了 {len(files_to_stitch)} 个视频块，将按数字顺序拼接。")
    print(f"重叠处理方式：前一个视频保留 {overlap_half} 帧，后一个视频保留 {overlap - overlap_half} 帧。")

    # --- 3. 准备写入器 ---
    output_path = os.path.join(input_dir, output_filename)
    try:
        first_video_path = files_to_stitch[0][1]
        with imageio.get_reader(first_video_path) as reader:
            fps = reader.get_meta_data().get('fps', 30)
    except Exception as e:
        print(f"错误：无法读取第一个视频块 '{first_video_path}' 的元数据。错误：{e}")
        return

    writer = imageio.get_writer(output_path, fps=fps)
    print(f"\n开始拼接视频... 输出到 '{output_path}' (FPS: {fps})")

    # --- 4. 逐个处理已排序的视频块并写入 (新的拼接逻辑) ---
    try:
        total_frames_written = 0
        num_files = len(files_to_stitch)
        progress_bar = tqdm(enumerate(files_to_stitch), total=num_files, desc="正在拼接", unit="file")
        
        for i, (chunk_num, video_path) in progress_bar:
            progress_bar.set_postfix_str(os.path.basename(video_path))
            reader = imageio.get_reader(video_path)
            num_frames_in_chunk = reader.count_frames()
            
            # ========================================================
            #  ↓↓↓ 这里是新的拼接逻辑 ↓↓↓
            # ========================================================
            
            # 确定要读取的起始帧
            # 第一个块从头开始，其他块跳过重叠部分的前半段
            start_frame = 0 if i == 0 else overlap_half

            # 确定要读取的结束帧
            # 最后一个块读到结尾，其他块在结尾处留出重叠部分的后半段
            is_last_chunk = (i == num_files - 1)
            end_frame = num_frames_in_chunk if is_last_chunk else (num_frames_in_chunk - (overlap - overlap_half))

            # ========================================================
            #  ↑↑↑ 逻辑结束 ↑↑↑
            # ========================================================

            # 逐帧读取并写入指定范围
            for frame_index in range(start_frame, end_frame):
                frame = reader.get_data(frame_index)
                writer.append_data(frame)
                total_frames_written += 1
            
            reader.close()
            
    except Exception as e:
        print(f"\n处理文件 '{video_path}' 时发生错误: {e}")
    finally:
        writer.close()
        print(f"\n拼接完成！总共写入 {total_frames_written} 帧。")


def main():
    parser = argparse.ArgumentParser(
        description="将文件夹中的视频块按文件名中的数字顺序拼接成一个完整的视频。"
    )
    
    parser.add_argument("input_dir", type=str, help="包含视频块的文件夹路径。")
    parser.add_argument("--output_filename", type=str, default="stitched_video.mp4", help="输出的完整视频的文件名。")
    parser.add_argument("--overlap", type=int, default=16, help="相邻视频块之间重叠的帧数。建议为偶数。")
    parser.add_argument("--exclude_suffix", type=str, default="_concat.mp4", help="拼接时要排除的文件后缀。")
    
    args = parser.parse_args()
    
    stitch_videos_by_number(
        input_dir=args.input_dir,
        output_filename=args.output_filename,
        overlap=args.overlap,
        exclude_suffix=args.exclude_suffix
    )

if __name__ == "__main__":
    main()

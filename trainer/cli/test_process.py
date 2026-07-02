import os
import json
import glob
import argparse
from typing import List, Dict, Optional

def prepare_test_data(
    video_base_dir: str,
    prompt_base_dir: str,
    output_json: str = "test_data.json",
    video_extensions: List[str] = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv'],
    recursive: bool = True
) -> None:
    """
    生成视频-提示词配对数据集
    
    参数:
    video_base_dir (str): 视频根目录路径
    prompt_base_dir (str): 提示词根目录路径
    output_json (str): 输出JSON文件路径
    video_extensions (List[str]): 支持的视频文件扩展名
    recursive (bool): 是否递归搜索子目录
    """
    # 验证输入目录
    if not os.path.isdir(video_base_dir):
        raise ValueError(f"视频目录不存在: {video_base_dir}")
    if not os.path.isdir(prompt_base_dir):
        raise ValueError(f"提示词目录不存在: {prompt_base_dir}")
    
    # 构建搜索模式
    search_pattern = os.path.join(video_base_dir, '**', '*') if recursive else os.path.join(video_base_dir, '*')
    
    # 收集所有视频文件
    video_files = []
    for ext in video_extensions:
        video_files.extend(glob.glob(
            f"{search_pattern}{ext}", 
            recursive=recursive
        ))
    
    if not video_files:
        raise ValueError(f"在 {video_base_dir} 中未找到视频文件（扩展名: {video_extensions}）")
    
    print(f"找到 {len(video_files)} 个视频文件")
    
    # 处理每个视频文件
    data_items = []
    unmatched_videos = []
    
    for video_path in video_files:
        # 获取视频文件名（无扩展名）
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        
        # 在提示词目录中查找匹配的txt文件
        prompt_path = os.path.join(prompt_base_dir, f"{video_name}.txt")
        
        if os.path.isfile(prompt_path):
            try:
                # 读取提示词内容（UTF-8编码）
                with open(prompt_path, 'r', encoding='utf-8') as f:
                    prompt = f.read().strip()
                
                # 验证提示词非空
                if not prompt:
                    print(f"警告: {prompt_path} 为空，跳过")
                    continue
                
                data_items.append({
                    "src_video_path": os.path.abspath(video_path),
                    "prompt": prompt
                })
            except Exception as e:
                print(f"处理 {prompt_path} 时出错: {str(e)}")
                continue
        else:
            unmatched_videos.append(video_path)
    
    # 输出统计信息
    print(f"\n处理完成:")
    print(f"- 成功匹配: {len(data_items)} 个视频")
    if unmatched_videos:
        print(f"- 未匹配: {len(unmatched_videos)} 个视频")
        for v in unmatched_videos[:5]:  # 只显示前5个
            print(f"  * {os.path.basename(v)}")
        if len(unmatched_videos) > 5:
            print(f"  ... 共 {len(unmatched_videos)} 个未匹配视频")
    
    # 保存JSON文件
    os.makedirs(os.path.dirname(os.path.abspath(output_json)), exist_ok=True)
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(data_items, f, indent=2, ensure_ascii=False)
    
    print(f"\n已生成数据集: {os.path.abspath(output_json)}")
    print(f"包含 {len(data_items)} 个有效样本")

def main():
    parser = argparse.ArgumentParser(
        description='视频-提示词数据集生成工具',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # 必需参数
    parser.add_argument('--video-dir', required=True, 
                        help='视频根目录路径（包含子文件夹）')
    
    parser.add_argument('--prompt-dir', required=True,
                        help='提示词根目录路径（包含同名txt文件）')
    
    # 可选参数
    parser.add_argument('--output', default='test_data.json',
                        help='输出JSON文件路径')
    
    parser.add_argument('--extensions', nargs='+', default=['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv'],
                        help='支持的视频扩展名（例如: .mp4 .avi）')
    
    parser.add_argument('--no-recursive', action='store_false', dest='recursive',
                        help='禁用子目录递归搜索')
    
    parser.add_argument('--verbose', action='store_true',
                        help='显示详细处理信息')
    
    args = parser.parse_args()
    
    # 调试信息
    if args.verbose:
        print("=== 参数配置 ===")
        print(f"视频目录: {args.video_dir}")
        print(f"提示词目录: {args.prompt_dir}")
        print(f"输出路径: {args.output}")
        print(f"视频扩展名: {args.extensions}")
        print(f"递归搜索: {args.recursive}")
        print("================")
    
    try:
        prepare_test_data(
            video_base_dir=args.video_dir,
            prompt_base_dir=args.prompt_dir,
            output_json=args.output,
            video_extensions=args.extensions,
            recursive=args.recursive
        )
    except Exception as e:
        print(f"❌ 处理失败: {str(e)}")
        exit(1)

if __name__ == "__main__":
    main()

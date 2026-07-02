import json
import argparse
import os

def duplicate_json_entries(input_json_path, output_json_path, prefix, duplication_count):
    """
    读取JSON文件，复制以指定prefix开头的条目n遍，并保存到新的JSON文件。

    Args:
        input_json_path (str): 输入的JSON文件的路径。
        output_json_path (str): 输出的新的JSON文件的路径。
        prefix (str): 用于匹配条目文件名的前缀。
        duplication_count (int): 匹配条目要复制的次数 (n)。
    """
    if not os.path.exists(input_json_path):
        print(f"错误：输入JSON文件 '{input_json_path}' 不存在。")
        return

    try:
        with open(input_json_path, 'r', encoding='utf-8') as f:
            original_dataset = json.load(f)
    except json.JSONDecodeError:
        print(f"错误：无法解析输入JSON文件 '{input_json_path}'。请确保它是有效的JSON格式。")
        return
    except Exception as e:
        print(f"错误：读取输入JSON文件 '{input_json_path}' 时发生错误: {e}")
        return

    print(f"成功加载输入JSON文件，包含 {len(original_dataset)} 条记录。")

    duplicated_entries = []
    matched_count = 0
    
    # 遍历原始数据集，查找匹配的条目并复制
    for item in original_dataset:
        # 获取文件名（不含路径和扩展名）
        # 这里假设 'video_path' 或 'src_video_path' 包含文件名
        # 我们使用 'video_path' 作为匹配依据，你可以根据需要调整
        # 如果是完整的绝对路径，需要进一步处理获取文件名
        # 假设 video_path 格式为 "dir/filename.ext"
        
        # 获取文件名，不含路径
        filename = os.path.basename(item.get("video_path", ""))
        
        # 获取文件名不含扩展名
        base_name, _ = os.path.splitext(filename)

        if base_name.startswith(prefix):
            matched_count += 1
            for _ in range(duplication_count):
                duplicated_entries.append(item)
        else:
            # 不匹配的条目也要添加到最终数据集中
            duplicated_entries.append(item)

    final_dataset = duplicated_entries
    
    print(f"匹配到 {matched_count} 条以 '{prefix}' 开头的条目。")
    print(f"每条复制 {duplication_count} 遍。")
    print(f"最终数据集将包含 {len(final_dataset)} 条记录。")

    # 将最终数据集写入新的JSON文件
    try:
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(final_dataset, f, ensure_ascii=False, indent=4)
        print(f"\n处理完成！修改后的数据集已保存至: '{output_json_path}'")
        print(f"最终数据集总条目: {len(final_dataset)}")
    except Exception as e:
        print(f"错误：写入输出JSON文件 '{output_json_path}' 失败: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="复制JSON文件中以特定前缀开头的条目n遍，用于平衡数据集类别。"
    )

    parser.add_argument(
        "-i", "--input-json",
        type=str,
        required=True,
        help="输入的JSON文件的路径。"
    )
    parser.add_argument(
        "-o", "--output-json",
        type=str,
        required=True,
        help="输出的新的JSON文件的路径。"
    )
    parser.add_argument(
        "-p", "--prefix",
        type=str,
        required=True,
        help="用于匹配条目文件名的前缀 (例如: 'Frozen')。"
    )
    parser.add_argument(
        "-n", "--count",
        type=int,
        required=True,
        help="匹配条目要复制的次数 (n)。"
    )

    args = parser.parse_args()

    duplicate_json_entries(
        input_json_path=args.input_json,
        output_json_path=args.output_json,
        prefix=args.prefix,
        duplication_count=args.count
    )

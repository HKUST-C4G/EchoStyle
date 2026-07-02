import os
import json
import argparse
import time
import threading
import concurrent.futures
from functools import partial, wraps
from http import HTTPStatus
import dashscope
from tqdm import tqdm
import random

# --- 全局配置 & 核心函数 (无改动) ---
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp')
VIDEO_EXTENSIONS = ('.mp4', '.mov', '.avi', '.mkv', '.flv', '.webm', '.ts')
SUPPORTED_EXTENSIONS = IMAGE_EXTENSIONS + VIDEO_EXTENSIONS
LOG_FILENAME = "processing_log.json"
log_lock = threading.Lock()

def retry(max_retries=3, wait_secs=1, exceptions=(Exception,)):
    # ... (代码不变)
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
                        print(f"达到最大重试次数后仍然失败: {func.__name__} with args {args}, kwargs {kwargs}. Error: {e}")
                        raise
                    time.sleep(wait_secs)
        return wrapper
    return decorator

@retry(max_retries=5, wait_secs=2)
def call_qwen_vl(file_path, prompt):
    # ... (代码不变)
    file_lower = file_path.lower()
    is_video = file_lower.endswith(VIDEO_EXTENSIONS)
    is_image = file_lower.endswith(IMAGE_EXTENSIONS)
    if is_video: media_key = "video"
    elif is_image: media_key = "image"
    else: raise ValueError(f"Unsupported file type: {file_path}")
    is_url = file_path.startswith("http")
    media_value = file_path if is_url else f"file://{os.path.abspath(file_path)}"
    messages = [{"role": "user", "content": [{media_key: media_value}, {"text": prompt}]}]
    response = dashscope.MultiModalConversation.call(model="qwen-vl-max", messages=messages)
    if response.status_code == HTTPStatus.OK: return response.output.choices[0].message.content[0]["text"]
    else:
        error_msg = f"Qwen-VL API Error for {file_path}: {response.code} - {response.message}"
        print(error_msg)
        raise Exception(error_msg)

def get_prompt(file_path, qwen_vl_prompt=None, verbose=True):
    # ... (代码不变)
    if qwen_vl_prompt is None: return None
    else:
        prompt = call_qwen_vl(file_path, qwen_vl_prompt)
        if verbose: print(f"Qwen-VL prompt for {os.path.basename(file_path)}: {prompt}")
        return prompt

def load_or_create_log(log_path):
    # ... (代码不变)
    if os.path.exists(log_path):
        with open(log_path, 'r', encoding='utf-8') as f:
            try: return json.load(f)
            except json.JSONDecodeError:
                print(f"警告: 日志文件 {log_path} 损坏，将创建新的日志。")
                return {}
    return {}

def save_log(log_path, log_data):
    # ... (代码不变)
    with log_lock:
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, indent=4, ensure_ascii=False)

def find_and_prepare_files(root_dir, log_data, sample_size=None):
    # ... (代码不变)
    print(f"正在扫描文件夹: {root_dir}...")
    all_files_on_disk = set()
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith(SUPPORTED_EXTENSIONS):
                all_files_on_disk.add(os.path.join(root, file))
    if sample_size is not None and sample_size > 0:
        print(f"\n--- 进入抽样模式，将随机选取 {sample_size} 个文件 ---")
        log_data.clear()
        file_list = list(all_files_on_disk)
        if len(file_list) <= sample_size:
            print(f"文件夹中文件总数 ({len(file_list)}) 小于或等于抽样数，将处理所有文件。")
            files_to_process = file_list
        else:
            files_to_process = random.sample(file_list, sample_size)
            print(f"已从 {len(file_list)} 个文件中随机抽取 {len(files_to_process)} 个。")
        for file_path in files_to_process:
            log_data[file_path] = {"status": "pending", "prompt": None, "error": None}
    else:
        for file_path in all_files_on_disk:
            if file_path not in log_data:
                log_data[file_path] = {"status": "pending", "prompt": None, "error": None}
        files_to_process = [path for path, data in log_data.items() if data['status'] in ["pending", "timeout", "error"] and os.path.exists(path)]
    return files_to_process, log_data

def process_single_file(file_path, qwen_prompt, log_data, log_path):
    # ... (代码不变)
    status, prompt_result, error_message = "unknown", None, None
    try:
        prompt_result = get_prompt(file_path, qwen_vl_prompt=qwen_prompt)
        if prompt_result is not None and "false" in prompt_result.lower(): status = "false"
        elif prompt_result is not None and "true" in prompt_result.lower(): status = "success"
        elif prompt_result is not None:
             status, error_message = "unknown_response", f"Unexpected response: {prompt_result}"
        else: status, error_message = "error", "get_prompt returned None"
    except concurrent.futures.TimeoutError: status, error_message = "timeout", "Processing timed out"
    except Exception as e: status, error_message = "error", str(e)
    log_data[file_path] = {"status": status, "prompt": prompt_result, "error": error_message}
    save_log(log_path, log_data)

# 1. 改造 main 函数以支持新模式
def main(args):
    log_path = os.path.join(args.folder_path, LOG_FILENAME)
    
    # --- 分支逻辑：是“仅删除”模式，还是完整处理模式 ---
    if args.delete_only:
        print("\n--- 进入“仅删除”模式 ---")
        print(f"将根据日志文件 '{log_path}' 中的记录执行删除操作。")
        
        final_log_data = load_or_create_log(log_path)
        if not final_log_data:
            print("错误: 未找到或日志文件为空，无法执行删除操作。")
            return
        
        # 在这种模式下，所有日志中的文件都是“待处理”的
        files_to_process = [] 
        
    else:
        # --- 完整处理模式（现有逻辑） ---
        if 'DASHSCOPE_API_KEY' not in os.environ:
            print("错误: 请先设置环境变量 DASHSCOPE_API_KEY。")
            return

        log_data = {} if args.sample_size else load_or_create_log(log_path)
        files_to_process, log_data = find_and_prepare_files(args.folder_path, log_data, sample_size=args.sample_size)
        save_log(log_path, log_data)

        if not files_to_process:
            print("所有文件均已处理完毕或没有文件需要处理。")
        else:
            print(f"共找到 {len(files_to_process)} 个文件需要处理...")
            qwen_prompt = args.qwen_prompt if args.qwen_prompt else "你是一名负责数据清洗的质检员。你收到的文件可能是视频或图片。如果视频或图像文件的画面不是片头或片尾(不含有大量的字幕文字)，且含有清晰的人物正脸，且无转场或十分剧烈的运动，则输出True。否则输出False。不要添加任何解释或额外文字。"
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
                process_func = partial(process_single_file, qwen_prompt=qwen_prompt, log_data=log_data, log_path=log_path)
                list(tqdm(executor.map(process_func, files_to_process), total=len(files_to_process), desc="处理媒体文件中"))
        
        # 处理完成后，重新加载最终日志
        final_log_data = load_or_create_log(log_path)


    # --- 统一的报告和清理部分 ---
    print("\n" + "="*50)
    print("处理完成！结果如下：")
    print("="*50)

    # 统计和报告逻辑 (完全复用)
    existing_paths = list(final_log_data.keys())
    total_files = len([p for p in existing_paths if os.path.exists(p)])
    false_files = [p for p in existing_paths if final_log_data.get(p, {}).get('status') == 'false' and os.path.exists(p)]
    timeout_files = [p for p in existing_paths if final_log_data.get(p, {}).get('status') == 'timeout' and os.path.exists(p)]
    error_files = [p for p in existing_paths if final_log_data.get(p, {}).get('status') in ['error', 'unknown_response'] and os.path.exists(p)]
    success_files = [p for p in existing_paths if final_log_data.get(p, {}).get('status') == 'success' and os.path.exists(p)]

    if not args.delete_only:
        print(f"本次运行处理的文件总数: {len(files_to_process)}")
    print(f"日志中记录的文件总数 (在磁盘上): {total_files}")
    print(f"成功处理 (True): {len(success_files)}")
    print(f"未通过检查 (False): {len(false_files)}")
    print(f"处理超时: {len(timeout_files)}")
    print(f"处理出错或响应未知: {len(error_files)}")
    
    # 详细列表打印 (复用，并增加详细信息)
    if timeout_files: print("\n--- 超时文件列表 ---"); [print(os.path.relpath(f, args.folder_path)) for f in timeout_files]
    if error_files: print("\n--- 出错文件/未知响应列表 ---"); [print(f"{os.path.relpath(f, args.folder_path)} -> 错误: {final_log_data[f].get('error', '未知')} | 原始响应: {final_log_data[f].get('prompt', '无')}") for f in error_files]
    if false_files: print("\n--- 标记为 'False' 的文件列表 ---"); [print(f"{os.path.relpath(f, args.folder_path)} -> 模型响应: {final_log_data[f].get('prompt', '无')}") for f in false_files]

    # 删除逻辑 (完全复用)
    files_to_delete = false_files + timeout_files + error_files
    if files_to_delete:
        print("\n" + "="*50 + "\n删除确认\n" + "="*50)
        try:
            confirm = input(f"共发现 {len(files_to_delete)} 个有问题的 (False, 出错, 超时) 文件。是否要全部删除? (y/n): ")
            if confirm.lower() == 'y':
                print("正在删除所有有问题的文件...")
                deleted_count = 0
                for file_to_delete in tqdm(files_to_delete, desc="删除有问题的 文件中"):
                    try:
                        os.remove(file_to_delete)
                        final_log_data[file_to_delete]['status'] = 'deleted'
                        deleted_count += 1
                    except OSError as e: print(f"\n删除失败: {file_to_delete} -> {e}")
                save_log(log_path, final_log_data) 
                print(f"成功删除了 {deleted_count} 个文件。")
            else: print("操作已取消，未删除任何文件。")
        except (KeyboardInterrupt, EOFError): print("\n操作已取消，未删除任何文件。")
    else: print("\n没有发现需要删除的有问题文件。")

    # 日志清理逻辑 (完全复用)
    if args.cleanup_log:
        print("\n" + "="*50)
        print("正在清理日志文件...")
        try:
            if os.path.exists(log_path):
                os.remove(log_path)
                print(f"日志文件 '{log_path}' 已成功删除。")
            else: print("日志文件不存在，无需清理。")
        except OSError as e: print(f"删除日志文件失败: {e}")

if __name__ == '__main__':
    # 2. 添加新的命令行参数
    parser = argparse.ArgumentParser(description="媒体文件质检与清理工具，支持全量/抽样处理、断点续传、仅删除等多种模式。")
    parser.add_argument("folder_path", type=str, help="包含媒体文件的根文件夹路径。")
    parser.add_argument("-p", "--qwen_prompt", type=str, default=None, help="（可选）自定义发送给 Qwen-VL 模型的提示词。")
    parser.add_argument("-w", "--workers", type=int, default=8, help="处理文件的线程数 (默认: 8)。")
    parser.add_argument("-n", "--sample-size", type=int, default=None, help="（可选）随机选取 n 个文件进行处理。如果提供此参数，将忽略已有的日志并进行全新扫描。")
    parser.add_argument("--cleanup-log", action="store_true", help="（可选）在程序执行结束后删除日志文件。")
    parser.add_argument(
        "--delete-only",
        action="store_true",
        help="（可选）仅删除模式。不调用API，直接根据现有日志文件执行删除操作。"
    )
    
    args = parser.parse_args()
    main(args)

import os
import json
import argparse
import time
import threading
import concurrent.futures
from functools import wraps
from http import HTTPStatus
import dashscope
from tqdm import tqdm # 用于显示进度条

# 确保已经设置了DASHSCOPE_API_KEY环境变量

# ==============================================================================
# 已有函数（原样保留）
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

    response = dashscope.MultiModalConversation.call(model="qwen-vl-max", messages=messages)
    if response.status_code == HTTPStatus.OK:
        return response.output.choices[0].message.content[0]["text"]
    else:
        raise Exception(f"Error: {response.code} - {response.message}")

# 注意：根据你的描述，get_prompt 函数在这里主要用于读取本地txt，
#      如果直接用Qwen-VL生成prompt，则会直接调用call_qwen_vl，
#      所以为了避免混淆，这里我们重新定义一个专门用于生成并存储prompt的函数。
#      原有的get_prompt函数在此场景下作用较小，或者需要修改其逻辑。
#      我们直接在worker函数中处理Qwen-VL调用和文件保存。

# ==============================================================================
# 新增和修改的代码
# ==============================================================================

def generate_and_save_prompt(image_file_path, output_dir, qwen_vl_base_prompt, overwrite_existing=False, verbose=True):
    """
    为单个图像文件生成prompt并存储。

    Args:
        image_file_path (str): 图像文件的完整路径。
        output_dir (str): prompt文本文件存储的目录。
        qwen_vl_base_prompt (str): 传递给Qwen-VL的基础prompt。
        overwrite_existing (bool): 如果为True，则覆盖已存在的prompt文件。
        verbose (bool): 是否打印详细信息。

    Returns:
        str or None: 如果成功生成并存储了prompt，返回生成的prompt文本；否则返回None。
    """
    # 构建输出txt文件的路径
    filename_without_ext = os.path.splitext(os.path.basename(image_file_path))[0]
    output_txt_path = os.path.join(output_dir, filename_without_ext + ".txt")

    if not overwrite_existing and os.path.exists(output_txt_path):
        if verbose:
            print(f"Skipping {os.path.basename(image_file_path)}: prompt file already exists at {output_txt_path}")
        return None # 跳过已存在的文件

    try:
        if verbose:
            print(f"Processing {os.path.basename(image_file_path)}...")

        # 调用Qwen-VL生成prompt
        generated_prompt = call_qwen_vl(image_file_path, qwen_vl_base_prompt)

        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)

        # 存储prompt到txt文件
        with open(output_txt_path, "w", encoding="utf-8") as f:
            f.write(generated_prompt)

        if verbose:
            print(f"Prompt for {os.path.basename(image_file_path)} saved to {output_txt_path}")
        return generated_prompt

    except Exception as e:
        print(f"Error processing {os.path.basename(image_file_path)}: {e}")
        return None

def process_images_in_folder(
    input_folder,
    output_folder,
    qwen_vl_base_prompt,
    num_workers=os.cpu_count(),
    overwrite_existing=False,
    verbose=True
):
    """
    处理输入文件夹中的所有图像文件，生成prompt并存储。

    Args:
        input_folder (str): 包含图像文件的输入目录。
        output_folder (str): prompt文本文件存储的目录。
        qwen_vl_base_prompt (str): 传递给Qwen-VL的基础prompt。
        num_workers (int): 用于并发处理的线程/进程数量。
        overwrite_existing (bool): 如果为True，则覆盖已存在的prompt文件。
        verbose (bool): 是否打印详细信息。
    """
    if not os.path.isdir(input_folder):
        print(f"错误：输入文件夹 '{input_folder}' 不存在。")
        return

    os.makedirs(output_folder, exist_ok=True) # 确保输出目录存在

    image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp") # 可以根据需要添加更多图像格式

    image_files_to_process = []
    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.lower().endswith(image_extensions):
                image_files_to_process.append(os.path.join(root, file))

    if not image_files_to_process:
        print(f"在 '{input_folder}' 中未找到支持的图像文件。")
        return

    print(f"在 '{input_folder}' 中找到 {len(image_files_to_process)} 个图像文件进行处理。")
    print(f"Prompt将存储到 '{output_folder}'。")
    print(f"使用 {num_workers} 个worker进行并发处理。")

    # 使用ThreadPoolExecutor实现多线程
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        # 使用tqdm包装map方法以显示进度条
        list(tqdm(executor.map(
            lambda img_path: generate_and_save_prompt(img_path, output_folder, qwen_vl_base_prompt, overwrite_existing, verbose),
            image_files_to_process
        ), total=len(image_files_to_process), desc="Generating Prompts"))

    print("\n所有图像文件的prompt生成和存储完成。")


# ==============================================================================
# 命令行参数解析 (使用argparse)
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="使用Qwen-VL为指定文件夹中的图像生成prompt并存储为txt文件。"
    )
    parser.add_argument(
        "input_folder",
        type=str,
        help="包含图像文件的输入文件夹路径。"
    )
    parser.add_argument(
        "output_folder",
        type=str,
        help="存储生成prompt的txt文件的输出文件夹路径。"
    )
    parser.add_argument(
        "--qwen_prompt",
        type=str,
        default="请详细描述图片内容，并避免描述风格。避免描述图标，水印，字幕，文字等无关内容。",
        help="传递给Qwen-VL模型的基础prompt。默认为'请详细描述图片内容，风格，颜色，主题等。'"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=os.cpu_count(),
        help=f"用于并发处理的线程数量。默认为CPU核心数 ({os.cpu_count()})。"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="如果指定此标志，将覆盖已存在的prompt文件。"
    )
    parser.add_argument(
        "--silent",
        action="store_true",
        help="如果指定此标志，将减少控制台输出，只显示进度条和关键错误。"
    )

    args = parser.parse_args()

    # 如果指定了 --silent，则将 verbose 设置为 False
    verbose_output = not args.silent

    # 调用主处理函数
    process_images_in_folder(
        input_folder=args.input_folder,
        output_folder=args.output_folder,
        qwen_vl_base_prompt=args.qwen_prompt,
        num_workers=args.workers,
        overwrite_existing=args.overwrite,
        verbose=verbose_output
    )

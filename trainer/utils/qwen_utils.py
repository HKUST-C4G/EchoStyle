import os
import time
from functools import wraps
from http import HTTPStatus

import dashscope

dashscope.api_key = os.environ.get("DASHSCOPE_API_KEY", "")


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


if __name__ == "__main__":
    print(call_qwen_vl("1.mp4", "详细描述这段视频内容,并且避免描述风格。"))
    print(call_qwen_vl("1.jpg", "详细描述这张图片"))

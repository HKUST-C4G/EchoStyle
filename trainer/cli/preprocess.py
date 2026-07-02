import argparse
import json
import os

from tqdm import tqdm

from trainer.utils import call_qwen_vl


def get_prompt(file_path, qwen_vl_prompt=None, verbose=True):
    """
    Get prompt from txt file or Qwen-VL.
    """
    if qwen_vl_prompt is None:
        txt_path = os.path.splitext(file_path)[0] + ".txt"
        if os.path.exists(txt_path):
            with open(txt_path, "r", encoding="utf-8") as f:
                return f.read().strip()
        else:
            filename = os.path.basename(file_path)
            print(f"Warning: {txt_path} not found, skipping {filename}")
            return None
    else:
        prompt = call_qwen_vl(file_path, qwen_vl_prompt)
        if verbose:
            print(f"Qwen-VL prompt: {prompt}")
        return prompt


def save_json(data, save_path):
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4, sort_keys=True)


def process_train_dataset(video_dir, output_json, qwen_vl_prompt=None):
    dataset = []
    video_extensions = [".mp4", ".avi", ".mov"]

    for video_file in tqdm(os.listdir(video_dir)):
        if any(video_file.lower().endswith(ext) for ext in video_extensions):
            video_path = os.path.join(video_dir, video_file)

            # Get prompt: from txt file if qwen_vl_prompt is None, otherwise use Qwen-VL.
            prompt = get_prompt(video_path, qwen_vl_prompt)
            if prompt is None:
                continue

            dataset.append({"video_path": video_path, "prompt": prompt})

    save_json(dataset, output_json)


def process_vace_train_dataset(video_dir, output_json, vace_modes, qwen_vl_prompt=None, vace_key_frame_index=None):
    dataset = []
    video_extensions = [".mp4", ".avi", ".mov"]
    image_extensions = [".jpg", ".jpeg", ".png", ".bmp"]

    for video_file in tqdm(os.listdir(video_dir)):
        if any(video_file.lower().endswith(ext) for ext in video_extensions):
            video_path = os.path.join(video_dir, video_file)

            # Get prompt: from txt file if qwen_vl_prompt is None, otherwise use Qwen-VL.
            prompt = get_prompt(video_path, qwen_vl_prompt)
            if prompt is None:
                continue

            # Build item with basic fields.
            item = {"video_path": video_path, "prompt": prompt, "vace_modes": vace_modes}

            if vace_key_frame_index is not None:
                item["vace_key_frame_index"] = vace_key_frame_index

            # Auto-detect reference images.
            video_basename = os.path.splitext(video_file)[0]
            vace_ref_image_paths = []

            # Look for reference images with pattern: {video_basename}_ref{N}.{ext}
            for file in os.listdir(video_dir):
                if file.startswith(f"{video_basename}_ref") and any(
                    file.lower().endswith(ext) for ext in image_extensions
                ):
                    ref_image_path = os.path.join(video_dir, file)
                    vace_ref_image_paths.append(ref_image_path)

            if vace_ref_image_paths:
                vace_ref_image_paths.sort()
                item["vace_ref_image_paths"] = vace_ref_image_paths

            dataset.append(item)

    save_json(dataset, output_json)

def process_vace_train_dataset_iv2v(video_dir, output_json, vace_modes, qwen_vl_prompt=None, vace_key_frame_index=None):
    dataset = []
    video_extensions = [".mp4", ".avi", ".mov"]
    image_extensions = [".jpg", ".jpeg", ".png", ".bmp"]

    for video_file in tqdm(os.listdir(video_dir)):
        if any(video_file.lower().endswith(ext) for ext in video_extensions):
            video_path = os.path.join(video_dir, video_file)

            # Get prompt: from txt file if qwen_vl_prompt is None, otherwise use Qwen-VL.
            prompt = get_prompt(video_path, qwen_vl_prompt)
            if prompt is None:
                continue

            # Build item with basic fields.
            item = {"video_path": video_path, "prompt": prompt, "vace_modes": vace_modes}

            if vace_key_frame_index is not None:
                item["vace_key_frame_index"] = vace_key_frame_index

            # Auto-detect reference images.
            video_basename = os.path.splitext(video_file)[0]
            vace_ref_image_paths = []

            # Look for reference images with pattern: {video_basename}_ref{N}.{ext}
            for file in os.listdir(video_dir):
                if file.startswith(f"{video_basename}_ref") and any(
                    file.lower().endswith(ext) for ext in image_extensions
                ):
                    ref_image_path = os.path.join(video_dir, file)
                    vace_ref_image_paths.append(ref_image_path)

            if vace_ref_image_paths:
                vace_ref_image_paths.sort()
                item["vace_ref_image_paths"] = vace_ref_image_paths

            dataset.append(item)

    save_json(dataset, output_json)

def process_i2v_eval_dataset(image_dir, output_json, qwen_vl_prompt=None):
    dataset = []
    image_extensions = [".jpg", ".jpeg", ".png", ".bmp"]

    for image_file in tqdm(os.listdir(image_dir)):
        if any(image_file.lower().endswith(ext) for ext in image_extensions):
            image_path = os.path.join(image_dir, image_file)

            # Get prompt: from txt file if qwen_vl_prompt is None, otherwise use Qwen-VL.
            prompt = get_prompt(image_path, qwen_vl_prompt)
            if prompt is None:
                continue

            dataset.append({"first_frame_path": image_path, "prompt": prompt})

    save_json(dataset, output_json)


def process_flf2v_eval_dataset(image_dir, output_json, qwen_vl_prompt=None):
    dataset = []

    # Find all first frame images (ending with _0).
    first_frame_files = []
    for image_file in os.listdir(image_dir):
        if any(image_file.endswith(f"_0.{ext}") for ext in ["jpg", "jpeg", "png"]):
            first_frame_files.append(image_file)

    for first_frame_file in tqdm(first_frame_files):
        # Get corresponding last frame file.
        # Use rsplit instead of replace("_0", "_1") to avoid replacing unintended "_0" substrings.
        # Example: "image_012_0.jpg" should become "image_012_1.jpg", not "image_112_1.jpg".
        base_name = first_frame_file.rsplit("_0.", 1)[0]
        ext = os.path.splitext(first_frame_file)[1]
        last_frame_file = f"{base_name}_1{ext}"

        first_frame_path = os.path.join(image_dir, first_frame_file)
        last_frame_path = os.path.join(image_dir, last_frame_file)

        # Check if last frame exists.
        if os.path.exists(last_frame_path):
            # Get prompt: from txt file if qwen_vl_prompt is None, otherwise use Qwen-VL on first frame.
            prompt = get_prompt(first_frame_path, qwen_vl_prompt)
            if prompt is None:
                continue

            item = {"first_frame_path": first_frame_path, "last_frame_path": last_frame_path, "prompt": prompt}
            dataset.append(item)

    save_json(dataset, output_json)


def process_vace_eval_dataset(image_dir, output_json, vace_modes, qwen_vl_prompt=None, vace_key_frame_index=None):
    dataset = []
    image_extensions = [".jpg", ".jpeg", ".png", ".bmp"]

    # Pre-filter files based on vace_modes.
    all_files = os.listdir(image_dir)
    if "flf2v" in vace_modes:
        # For flf2v mode, only process *_0 images (first_frame).
        image_files = [f for f in all_files if any(f.endswith(f"_0.{ext}") for ext in ["jpg", "jpeg", "png"])]
    else:
        # For other modes, process all image files.
        image_files = [f for f in all_files if any(f.lower().endswith(ext) for ext in image_extensions)]

    for image_file in tqdm(image_files):
        image_path = os.path.join(image_dir, image_file)

        # Get prompt: from txt file if qwen_vl_prompt is None, otherwise use Qwen-VL.
        prompt = get_prompt(image_path, qwen_vl_prompt)
        if prompt is None:
            continue

        # Build item with basic fields.
        item = {"prompt": prompt, "vace_modes": vace_modes}

        # Set appropriate frame path based on vace_modes.
        if "kf2v" in vace_modes:
            item["key_frame_path"] = image_path
        elif "lf2v" in vace_modes:
            item["last_frame_path"] = image_path
        elif "flf2v" in vace_modes:
            # For flf2v mode, set both first_frame_path and last_frame_path.
            item["first_frame_path"] = image_path

            # Find corresponding last frame (*_1).
            base_name = image_file.rsplit("_0.", 1)[0]
            ext = os.path.splitext(image_file)[1]
            last_frame_file = f"{base_name}_1{ext}"
            last_frame_path = os.path.join(image_dir, last_frame_file)

            if os.path.exists(last_frame_path):
                item["last_frame_path"] = last_frame_path
            else:
                raise FileNotFoundError(f"Last frame not found for {image_file}")
        else:
            # Default to first_frame_path for i2v.
            item["first_frame_path"] = image_path

        if vace_key_frame_index is not None:
            item["vace_key_frame_index"] = vace_key_frame_index

        # Auto-detect reference images.
        image_basename = os.path.splitext(image_file)[0]
        vace_ref_image_paths = []

        # Look for reference images with pattern: {image_basename}_ref{N}.{ext}
        for file in os.listdir(image_dir):
            if file.startswith(f"{image_basename}_ref") and any(
                file.lower().endswith(ext) for ext in image_extensions
            ):
                ref_image_path = os.path.join(image_dir, file)
                vace_ref_image_paths.append(ref_image_path)

        if vace_ref_image_paths:
            vace_ref_image_paths.sort()
            item["vace_ref_image_paths"] = vace_ref_image_paths

        dataset.append(item)

    save_json(dataset, output_json)


def main():
    parser = argparse.ArgumentParser(description="Process dataset for LoRA training and evaluation.")
    parser.add_argument(
        "--dataset_type",
        choices=["train", "vace_train", "i2v_eval", "flf2v_eval", "vace_eval"],
        required=True,
        help="Type of dataset to process.",
    )
    parser.add_argument(
        "--lora_type",
        help=(
            "Type of LoRA to process. Determines how prompts are generated: if not provided, prompts will be read "
            "from txt files; if provided, uses Qwen-VL model to generate prompts based on the corresponding template "
            "from `prompt_template.json`."
        ),
    )
    parser.add_argument("--input_dir", required=True, help="Input directory path.")
    parser.add_argument("--output_json", help="Output JSON file path. Default: input_dir/metadata.json")
    parser.add_argument(
        "--vace_modes",
        nargs="+",
        choices=["i2v", "flf2v", "lf2v", "kf2v", "ref_images"],
        help="VACE modes for vace_train and vace_eval datasets. Can specify multiple modes.",
    )
    parser.add_argument(
        "--vace_key_frame_index",
        type=int,
        help="Key frame index for VACE kf2v mode. Required when using kf2v mode.",
    )

    args = parser.parse_args()

    # Validate vace_modes requirement for vace_train and vace_eval.
    if args.dataset_type in ["vace_train", "vace_eval"] and args.vace_modes is None:
        parser.error(f"--vace_modes is required when dataset_type is '{args.dataset_type}'")

    # Set default output_json if not provided.
    if args.output_json is None:
        args.output_json = os.path.join(args.input_dir, "metadata.json")
        print(f"Output JSON file path not provided, using default: {args.output_json}")

    qwen_vl_prompt = None
    if args.lora_type is not None:
        prompt_template = json.load(open("prompt_template.json"))
        qwen_vl_prompt = prompt_template["templates"][args.lora_type]
        if args.dataset_type == "train":
            qwen_vl_prompt = prompt_template["train_prefix"] + qwen_vl_prompt
        else:
            qwen_vl_prompt = prompt_template["eval_prefix"] + qwen_vl_prompt

    if args.dataset_type == "train":
        process_train_dataset(args.input_dir, args.output_json, qwen_vl_prompt)
    elif args.dataset_type == "vace_train":
        process_vace_train_dataset(
            args.input_dir, args.output_json, args.vace_modes, qwen_vl_prompt, args.vace_key_frame_index
        )
    elif args.dataset_type == "i2v_eval":
        process_i2v_eval_dataset(args.input_dir, args.output_json, qwen_vl_prompt)
    elif args.dataset_type == "flf2v_eval":
        process_flf2v_eval_dataset(args.input_dir, args.output_json, qwen_vl_prompt)
    elif args.dataset_type == "vace_eval":
        process_vace_eval_dataset(
            args.input_dir, args.output_json, args.vace_modes, qwen_vl_prompt, args.vace_key_frame_index
        )


if __name__ == "__main__":
    main()

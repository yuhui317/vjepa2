import os
import subprocess
import numpy as np
import torch
from decord import VideoReader
from transformers import AutoModel, AutoVideoProcessor

def get_video():
    vr = VideoReader("sample_video.mp4")
    # 只抽取更少的帧来降低内存和CPU负担
    frame_idx = np.arange(0, 32, 2)
    video = vr.get_batch(frame_idx).asnumpy()
    return video

def main():
    hf_model_name = "facebook/vjepa2-vitl-fpc64-256" # ViT-Large (300M)
    sample_video_path = "sample_video.mp4"
    if not os.path.exists(sample_video_path):
        video_url = "https://huggingface.co/datasets/nateraw/kinetics-mini/resolve/main/val/bowling/-WH-lxmGJVY_000005_000015.mp4"
        subprocess.run(["wget", video_url, "-O", sample_video_path])
        print("Downloading video")

    print(f"Loading {hf_model_name} on CPU...")
    # 强制在 CPU 上运行
    device = "cpu"
    model_hf = AutoModel.from_pretrained(hf_model_name)
    model_hf.to(device)
    model_hf.eval()

    hf_transform = AutoVideoProcessor.from_pretrained(hf_model_name)
    print("Loading video...")
    video = get_video()
    video = torch.from_numpy(video).permute(0, 3, 1, 2)

    print("Preprocessing...")
    x_hf = hf_transform(video, return_tensors="pt")["pixel_values_videos"].to(device)

    print("Extracting features (This might take a moment on CPU)...")
    with torch.inference_mode():
        out_patch_features_hf = model_hf.get_vision_features(x_hf)

    print(f"Success! Extracted features shape: {out_patch_features_hf.shape}")

if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
# @FileName: DepthAnythingV2.py

import os
import cv2
import torch
import time
import argparse
import json
import numpy as np
import imageio
from typing import List, Dict
from pathlib import Path

from depth_anything_v2.depth_anything_v2.dpt import DepthAnythingV2 as DA2

class DepthInferenceEngine:
    """
    负责深度估计模型的加载、预热以及核心推理与彩色化处理
    """
    def __init__(self, encoder='vits', device=None):
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
        self.encoder_type = encoder
        
        self.model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
        }
        
        print(f"[*] [Engine] 正在加载模型: {encoder} | 运行设备: {self.device}")
        self.model = DA2(**self.model_configs[encoder])
        
        ckpt_path = os.path.join(Path(__file__).resolve().parent, f"checkpoints/depth_anything_v2_{encoder}.pth")

        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"未找到模型权重文件: {ckpt_path}")
            
        self.model.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
        self.model = self.model.to(self.device).eval()
        
    def warmup(self):
        """预热推理"""
        print("[*] [Engine] 正在进行 GPU 预热...")
        dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
        with torch.no_grad():
            _ = self.model.infer_image(dummy_img)
        print("[*] [Engine] 预热完成.")

    @torch.no_grad()
    def predict_visual(self, image: np.ndarray) -> np.ndarray:
        """
        推理并应用 VIRIDIS 映射 (OpenCV 的 BGR 格式输出)
        """
        depth = self.model.infer_image(image)
        depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-5) * 255.0
        depth_norm = depth_norm.astype(np.uint8)
        # 生成 BGR 格式彩色图
        color_depth = cv2.applyColorMap(depth_norm, cv2.COLORMAP_VIRIDIS)
        return color_depth

class PerformanceMonitor:
    """
    负责记录处理指标并导出 JSON 评估结果
    """
    def __init__(self, fps_config: int, model_name: str, device: str):
        self.latencies = []
        self.image_size = (0, 0)
        self.fps_config = fps_config
        self.model_name = model_name
        self.device = device

    def record(self, duration: float, width: int, height: int):
        self.latencies.append(duration)
        if self.image_size == (0, 0):
            self.image_size = (width, height)

    def get_summary_dict(self) -> Dict:
        avg_lat = np.mean(self.latencies) if self.latencies else 0
        total_time = np.sum(self.latencies)
        return {
            "model_info": {
                "encoder": self.model_name,
                "device": self.device,
                "input_fps_config": self.fps_config
            },
            "image_info": {
                "width": self.image_size[0],
                "height": self.image_size[1]
            },
            "benchmarks": {
                "total_frames": len(self.latencies),
                "total_inference_time_s": round(float(total_time), 4),
                "avg_latency_ms": round(float(avg_lat * 1000), 2),
                "throughput_fps": round(float(1.0 / avg_lat), 2) if avg_lat > 0 else 0
            }
        }

class DataManager:
    """
    负责文件扫描、单帧保存、视频/GIF 合成及 JSON 导出
    """
    def __init__(self, mps_path: str):
        self.root = mps_path
        self.input_base = os.path.join(self.root, "aria/all_data")
        self.output_base = os.path.join(self.root, "aria/all_data")
        self.da2_root = os.path.join(self.root, "aria")
        
        # 确保目录存在
        os.makedirs(self.output_base, exist_ok=True)
        
        self.video_path = os.path.join(self.da2_root, "da2_video_vis.mp4")
        self.gif_path = os.path.join(self.da2_root, "da2_video_vis.gif")
        self.json_path = os.path.join(self.da2_root, "da2_results.json")
        
        self.video_writer = None
        self.gif_frames = [] # GIF 需要缓存所有帧或使用特定 writer

    def get_folders(self) -> List[str]:
        return sorted([f for f in os.listdir(self.input_base) if os.path.isdir(os.path.join(self.input_base, f))])

    def save_single_image(self, bgr_img: np.ndarray, folder: str):
        target = os.path.join(self.output_base, folder)
        os.makedirs(target, exist_ok=True)
        cv2.imwrite(os.path.join(target, "depth.png"), bgr_img)

    def init_media_writers(self, w: int, h: int, fps: int):
        # 初始化 VideoWriter (MP4)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(self.video_path, fourcc, fps, (w, h))
        print(f"[*] [IO] 视频与 GIF 写入器初始化成功 (FPS: {fps})")

    def add_frame(self, bgr_img: np.ndarray):
        # 写入 MP4
        if self.video_writer:
            self.video_writer.write(bgr_img)
        # 缓存 GIF 帧 (GIF 通常需要 RGB 格式)
        rgb_frame = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        self.gif_frames.append(rgb_frame)

    def finalize(self, metrics: Dict, fps: int):
        # 1. 释放视频
        if self.video_writer:
            self.video_writer.release()
        
        # 2. 生成 GIF
        print(f"[*] [IO] 正在合成 GIF (共 {len(self.gif_frames)} 帧)...")
        imageio.mimsave(self.gif_path, self.gif_frames, fps=fps, loop=0)
        
        # 3. 保存 JSON
        with open(self.json_path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=4, ensure_ascii=False)
        
        print(f"[*] [IO] 媒体文件已保存至: {self.da2_root}")

class DepthAnythingV2:
    def __init__(self, mps_path: str, fps: int = 0):
        self.engine = DepthInferenceEngine()
        self.data = DataManager(mps_path)
        self.monitor = PerformanceMonitor(fps, self.engine.encoder_type, self.engine.device)
        
        if fps == 0: # 未赋值（从文件里读取）
            cam_config_json_path = os.path.join(mps_path,'aria','aria_cam_config.json')
            if os.path.exists(cam_config_json_path):
                try:
                    with open(cam_config_json_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    fps = data.get("fps")
                except (json.JSONDecodeError, IOError):
                    # 如果文件损坏、为空或读取失败，设为默认值
                    fps = 10 # default
            else:
                # 如果文件不存在，直接设为默认值
                fps = 10 # default
        self.fps = fps

    def run(self):
        self.engine.warmup()
        folders = self.data.get_folders()
        print(f"[*] [App] 开始处理，总计 {len(folders)} 个目录")

        for folder in folders:
            rgb_path = os.path.join(self.data.input_base, folder, "rgb.png")
            if not os.path.exists(rgb_path): continue
            
            raw_bgr = cv2.imread(rgb_path)
            if raw_bgr is None: continue

            # 执行推理并统计
            t0 = time.perf_counter()
            color_bgr = self.engine.predict_visual(raw_bgr)
            if torch.cuda.is_available(): torch.cuda.synchronize()
            t1 = time.perf_counter()

            # 首次运行初始化媒体写入
            if self.data.video_writer is None:
                h, w = color_bgr.shape[:2]
                self.data.init_media_writers(w, h, self.fps)

            # 存储与记录
            self.data.save_single_image(color_bgr, folder)
            self.data.add_frame(color_bgr)
            self.monitor.record(t1 - t0, color_bgr.shape[1], color_bgr.shape[0])
            
            print(f"  > [Processed] {folder} | { (t1-t0)*1000:.1f}ms")

        # 结束并导出
        summary = self.monitor.get_summary_dict()
        self.data.finalize(summary, self.fps)
        
        # 专业控制台输出
        print("\n" + "═"*60)
        print("                处理任务成功完成")
        print("═"*60)
        print(f" • 视频保存路径 : {self.data.video_path}")
        print(f" • GIF 保存路径  : {self.data.gif_path}")
        print(f" • 数据报告路径 : {self.data.json_path}")
        print(f" • 平均帧率     : {summary['benchmarks']['throughput_fps']} FPS")
        print("═"*60 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mps_path", type=str, required=True, help="根目录路径")
    args = parser.parse_args()

    depth_anything_v2 = DepthAnythingV2(args.mps_path)
    depth_anything_v2.run()

# conda activate aria
# cd src
# python -m depth_anything_v2.DepthAnythingV2 --mps_path "./data/mps_open_cabinet_5_vrs/" 
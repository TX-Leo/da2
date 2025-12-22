import cv2
import torch
import time
from depth_anything_v2.dpt import DepthAnythingV2

# --- 1. 设置与加载 ---
DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f"Running on device: {DEVICE}")

model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}

encoder = 'vits' # 使用大模型
print(f"Loading model {encoder}...")
model = DepthAnythingV2(**model_configs[encoder])
model.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_{encoder}.pth', map_location='cpu'))
model = model.to(DEVICE).eval()

# --- 2. 准备数据 ---
image_path = 'door.jpg'
target_size = 1408 # 目标分辨率

# --- 3. 预热 (Warm-up) ---
print("Warming up...")
raw_img_warmup = cv2.imread(image_path)
with torch.no_grad():
    # 预热时也使用 target_size，确保 GPU 显存分配是针对 224 的
    _ = model.infer_image(raw_img_warmup, input_size=target_size)

# --- 4. 正式测速 ---
num_runs = 50
print(f"Starting timing over {num_runs} runs with Input Size {target_size}x{target_size}...")

io_times = []      # 磁盘读取
resize_times = []  # 图片缩放
infer_times = []   # 模型推理

if DEVICE == 'cuda':
    torch.cuda.synchronize()

for _ in range(num_runs):
    with torch.no_grad():
        # --- A. 磁盘 IO ---
        t0 = time.perf_counter()
        raw_img = cv2.imread(image_path)
        t1 = time.perf_counter()
        
        # --- B. 显式 Resize (CPU) ---
        # 注意：虽然 infer_image 内部也能做 resize，但在机器人流程中，
        # 你可能想先 resize 再做其他处理，所以这里单独测这个步骤
        resized_img = cv2.resize(raw_img, (target_size, target_size))
        t2 = time.perf_counter()
        
        # --- C. 模型推理 (GPU) ---
        # 关键点：必须传入 input_size=224，否则它会默认 resize 到 518！
        # 这里的 resized_img已经是224了，传入 input_size=224 可以避免内部再次 resize
        depth = model.infer_image(resized_img, input_size=target_size)
        
        if DEVICE == 'cuda':
            torch.cuda.synchronize()
        t3 = time.perf_counter()
        
        io_times.append(t1 - t0)
        resize_times.append(t2 - t1)
        infer_times.append(t3 - t2)

# --- 5. 结果计算 ---
avg_io = sum(io_times) / num_runs
avg_resize = sum(resize_times) / num_runs
avg_infer = sum(infer_times) / num_runs
avg_total = avg_io + avg_resize + avg_infer
fps = 1 / avg_total

print(f"==================================================")
print(f"Config: {encoder} @ {target_size}x{target_size}")
print(f"==================================================")
print(f"1. Image Load (IO):      {avg_io*1000:.2f} ms")
print(f"2. CV2 Resize (CPU):     {avg_resize*1000:.2f} ms")
print(f"3. Inference (GPU):      {avg_infer*1000:.2f} ms")
print(f"--------------------------------------------------")
print(f"Total Latency:           {avg_total*1000:.2f} ms")
print(f"FPS:                     {fps:.2f}")
print(f"==================================================")


# (DA2) ➜  Depth-Anything-V2 git:(main) ✗ python demo.py
# xFormers not available
# xFormers not available
# Running on device: cuda
# Loading model vits...
# Warming up...
# Starting timing over 50 runs with Input Size 224x224...
# ==================================================
# Config: vits @ 224x224
# ==================================================
# 1. Image Load (IO):      18.32 ms
# 2. CV2 Resize (CPU):     0.10 ms
# 3. Inference (GPU):      3.56 ms
# --------------------------------------------------
# Total Latency:           21.99 ms
# FPS:                     45.48


# 我现在想写一个脚本，来使用depth anything把每一个aria生成的rgb照片都获取一下depth信息，但是有个问题在于我的我想要绝对指标，因为depth anything生成的结果不是metric的
# 我用一些方法获得了一个物体（handle）的一些keypoints的点（包括3d点和2d点），其中包含handle静止和运动的时候
# 请你帮我使用这个来写一个脚本来获取metric的depth image（我会当作一个channel和rgb的3个channel融合）


# 遍历"../data/mps_open_cabinet_5_vrs/aria/all_data/"下的所有图片，这个文件夹下的结果如图所示，每一个都是XXXXX文件夹，请你参考我提供给你demo.py代码帮我用depthaything运行每一个图片，就原始图片大小就行，不用resize，然后生成一个1*image_w*image_h的图片，并且保存到"../data/mps_open_cabinet_5_vrs/da2/all_data/"，
# 当然这个文件夹可能不存在，注意代码鲁棒性，而且也是这种"XXXXX/depth.png"这种格式，还有我print运行每一张图片的耗时，然后最后输出整体运行的一些速度指标（自己考虑用什么合适），记得print界面应该表现的很专业且易读
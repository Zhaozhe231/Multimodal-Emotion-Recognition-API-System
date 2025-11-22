"Author: Zhaozhe Zhang"

from hailo_sdk_client import ClientRunner
import os
import cv2
import numpy as np

##############################
# 1. 参数配置（根据实际情况修改）
##############################
# 模型相关
onnx_path = "/home/user/facial_emotion_detection/face_emotion_model.onnx"  # ONNX 文件路径
model_name = "face_emotion_model"                                         # 模型名称，可自定义
start_node = "input"      # 模型输入节点名称
end_node   = "dense_1"    # 模型输出节点名称
input_shape = [1, 1, 48, 48]  # Batch=1, 通道=1, 高=48, 宽=48（示例：灰度输入）
hw_arch = "hailo8l"       # 目标硬件架构，如 Hailo-8L、Hailo-8、Hailo-15H 等

# 文件名
har_path         = f"{model_name}_native.har"      # 解析后的 HAR 文件
quant_har_path   = f"{model_name}_quantized.har"   # 量化后的 HAR 文件
hef_path         = f"{model_name}.hef"             # 最终编译生成的 HEF 文件

# 校准数据集
calib_images_dir = "/home/user/facial_emotion_detection/fer2013"  # 用于量化的校准图片文件夹
calib_count      = 1000  # 使用多少张图片进行量化校准
img_height, img_width = 48, 48  # 与 input_shape 中的高、宽对应

##############################
# 2. 解析 ONNX 模型 => HAR
##############################
runner = ClientRunner(hw_arch=hw_arch)

# translate_onnx_model：将 ONNX 转换为内部表示
hn, npz = runner.translate_onnx_model(
    model=onnx_path,
    net_name=model_name,
    start_node_names=[start_node],
    end_node_names=[end_node],
    net_input_shapes={start_node: input_shape}
)
runner.save_har(har_path)
print(f"Successfully saved native HAR to: {har_path}")

##############################
# 3. 准备校准数据集
##############################
# 从 fer2013 文件夹中读取若干张图像，用于量化过程
images_list = [img_name for img_name in os.listdir(calib_images_dir)
               if os.path.splitext(img_name)[1].lower() in [".jpg", ".jpeg", ".png", ".bmp"]]
images_list = images_list[:calib_count]  # 截取前 calib_count 张

# 初始化校准数组：形状 (N, H, W, C)
calib_dataset = np.zeros((len(images_list), img_height, img_width, 1), dtype=np.uint8)  # 通道=1

for idx, img_name in enumerate(sorted(images_list)):
    img_path = os.path.join(calib_images_dir, img_name)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # 灰度读取
    if img is None:
        continue
    resized = cv2.resize(img, (img_width, img_height))
    # 扩展维度到 (H, W, 1)
    resized_3d = np.expand_dims(resized, axis=-1)
    calib_dataset[idx] = resized_3d

##############################
# 4. 量化优化 => 量化后的 HAR
##############################
runner = ClientRunner(har=har_path)

# 通过模型脚本命令设置一些优化/性能参数，示例：
alls_lines = [
    "model_optimization_flavor(optimization_level=1, compression_level=2)",
    "resources_param(max_control_utilization=0.6, max_compute_utilization=0.6, max_memory_utilization=0.6)",
    "performance_param(fps=10)"
]
runner.load_model_script('\n'.join(alls_lines))

runner.optimize(calib_dataset)
runner.save_har(quant_har_path)
print(f"Successfully saved quantized HAR to: {quant_har_path}")

##############################
# 5. 编译 => 生成 HEF 文件
##############################
runner = ClientRunner(har=quant_har_path)
compiled_hef = runner.compile()

with open(hef_path, "wb") as f:
    f.write(compiled_hef)

print(f"Successfully compiled HEF file: {hef_path}")

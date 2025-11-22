"Author: Zhaozhe Zhang"

import tensorflow as tf
import tf2onnx

# 加载 Keras 模型
model = tf.keras.models.load_model('face_emotion_model.h5')

# 如果模型没有 output_names 属性，则手动添加
if not hasattr(model, 'output_names'):
    model.output_names = [model.layers[-1].name]

# 定义模型输入签名（这里假定输入为 48x48 的灰度图像）
spec = (tf.TensorSpec((None, 48, 48, 1), tf.float32, name="input"),)

# 转换为 ONNX 模型并保存
output_path = "face_emotion_model.onnx"
model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13, output_path=output_path)
print("转换完成，ONNX 模型保存在:", output_path)

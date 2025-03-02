import io
import json
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, jsonify, request
# onnxruntime 是cpu版本的，想使用GPU，还需要研究一下
import onnxruntime
from gevent import monkey

monkey.patch_all()
from gevent import pywsgi

app = Flask(__name__)
# 加载类别字典
imagenet_class_index = json.load(open('imagenet_class_index.json'))


"""
注意：当前是使用 onnx模型格式 

查看网络结构：
    可直接在：https://netron.app/ 网址上查看自己训练后保存的onnx格式模型 的内部结构

"""
# 使用onnx模型格式加载
ort_session = onnxruntime.InferenceSession("cnn.onnx")


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def transform_image(image_bytes):
    """
    图片预处理
    :param image_bytes: 二进制的图像
    :return: tensor格式的图片
    """
    my_transforms = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                            [0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])])
    image = Image.open(io.BytesIO(image_bytes))
    return my_transforms(image).unsqueeze(0)


def get_prediction(image_bytes):
    """
    模型预测
    :param image_bytes: tensor类型图像
    :return: 预测结果
    """
    tensor = transform_image(image_bytes=image_bytes)

    # 使用onnx加载
    # 先转为np格式
    img1 = np.array(tensor)
    # onnx模型小套餐~
    ortvalue = onnxruntime.OrtValue.ortvalue_from_numpy(img1)
    ort_inputs = {ort_session.get_inputs()[0].name: ortvalue}
    ort_outs = ort_session.run(None, ort_inputs)

    # 求列表内类别概率最大索引
    a1 = ort_outs[0][0].argmax()
    # 数据需要字符串格式
    predicted_idx = str(a1)
    return imagenet_class_index[predicted_idx]


# /predict路径对应客户端路径
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        # 读取数据
        img_bytes = file.read()
        # 预测
        class_name = get_prediction(image_bytes=img_bytes)
        # 返回预测结果
        return jsonify(class_name)


if __name__ == '__main__':
    # 使用猴子补丁(协程支持)实现高并发
    # 指定服务器ip地址与端口号
    server = pywsgi.WSGIServer(('192.168.3.11', 5001), app)
    server.serve_forever()
    # 注意：
    # gevent 的使用并不能减少实际 CPU 的使用量，所以若程序的执行过程消耗的全为 CPU 资源，则其异步也是毫无意义的。

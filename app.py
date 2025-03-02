import io
import json
import torch
from vgg19 import VGG19
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, jsonify, request

from gevent import monkey
monkey.patch_all()
from gevent import pywsgi


app = Flask(__name__)
# 加载类别字典
imagenet_class_index = json.load(open('imagenet_class_index.json'))
# 加载自定义模型
model = VGG19([16, 'Max', 30, 51, 'Max',
               62, 103, 96, 101, 'Max',
               92, 171, 150, 154, 'Max',
               124, 142, 140, 65, 'Max'])
# 加载权重
model.load_state_dict(torch.load("vgg16_16_0.94_acc_max.pth"), strict=True)
# 预测开关 
model.eval()


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
    outputs = model.forward(tensor)
    _, y_hat = outputs.max(1)
    predicted_idx = str(y_hat.item())
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

import numpy as np
import requests
import json
import time
import os


def ke_hu1(path):
    a = time.time()
    try:
        # IP对应，/predict路径也要对应
        resp = requests.post("http://192.168.3.11:5001/predict",
                             files={"file": open(path, 'rb')})

        if resp.status_code == 200:
            class_ = json.loads(resp.content)

            b = time.time() - a

            return class_, b
    except:
        print("错了")
        return None, None


# 指定访问的图片
path = "./Cat"
# 图像列表
imgs = os.listdir(path)
# 访问时间列表 
time_list = []
for i in imgs:
    class_n, time_n = ke_hu1(path + "/" + i)
    if class_n:
        time_list.append(time_n)

if time_list:
    print(time_list)
    sum = np.sum(time_list) / len(time_list)
    print("平均耗时{:.2f}".format(sum))
    print("最高耗时{:.2f}".format(np.max(time_list)))
    print("最低耗时{:.2f}".format(np.min(time_list)))

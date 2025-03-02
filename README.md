猫狗分类项目部署

使用的模型是剪枝后的猫狗分类项目

基于上一个环境增加两个库

pip install flask

pip install gevent



基于flask + gevent部署方案，适用于小吞吐量级

将训练好的模型加载，服务端开启服务,客户端访问服务器，返回识别结果。就这么简单。

运行app.py

示例模型权重链接: https://pan.baidu.com/s/1xCGoOFIZqtF1kibdqYP6WA  密码: 6kdc


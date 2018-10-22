## **ECO_Full_kinetics_pretrained**

____

### dataset
____
1. 数据格式为解码视频文件后保存的每一帧的numpy数组文件，即.npy文件
2. 每个视频对应一个.npy文件，平均分段截取16帧，读取每个视频的.npy文件后的numpy数组形状为(16, 3, h, w), 如(16, 3, 1280, 720)
3. 解码后的.npy文件存放路径为/data/jh/notebooks/hudengjun/meitu/ECOFrames/

### loss
____
loss的实现知识针对单标签的SoftmaxCrossEntropyLoss

### training 
____
1. train_symbol_net.py训练过程中创建symbol网络，或者通过ECO预训练的.json和.params文件创建网络，示例代码：
```bash
$ python train_symbol_net.py --lr 0.1 --batch-size 16 --gpus 0,1
```

2. train_gluon_net.py训练过程中创建gluon实现的ECO_Full网络，示例代码：
```bash
$ python train_gluon_net.py --lr 0.1 --batch-size 16 --gpus 0,1
```

### todo list
____
1. 多标签loss的实现
2. 训练优化网络
3. 优化项目代码结构

### note
____
1. pretrained_models/ 保存了预训练网络的.json和.params文件
2. model/ 保存训练过程中网络的训练权重，用于中断训练后的恢复
3. data/ 保存数据处理程序
4. network/ 保存网络实现程序

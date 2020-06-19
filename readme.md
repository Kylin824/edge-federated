###操作步骤：以nniid数据集为例

前期准备:
1. 数据集解压到当前project目录中，目录名默认为/client_dataset/mnist/nniid/
2. 新建目录存放训练结果 /result/mnist/nniid/

运行准备:
1. FL_Server.py 
    - 修改ip:port
    - 设置训练轮次：MAX_NUM_ROUNDS = 50 
2. FL_Client.py 
    - 修改ip:port 
    - 修改186行 " if (property >= 0.7): "中的数值  例如0.7代表3个client都有70%存活 30%概率宕机
3. datasource.py
    - 修改data_dir：设置client读取的数据集目录为/client_dataset/mnist/nniid/
    
运行方式：
1. 先启动FL_Server.py
2. 分别启动FL_Client.py，三个client的index参数设置为0, 1, 2
    
运行结束：
/result/mnist/nniid/目录中会生成4个txt文件，分别为3个client各自的accuracy，以及global model的accuracy
通过plot_result.py中的plot_acc()函数，填入acc文件位置即可画图展示
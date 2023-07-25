import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Optional

def save_lossplot(path: Optional[str],
                  name: Optional[str]):

    ckpt_path = path

    for f in os.listdir(ckpt_path):
        if '_stats' in f:
            file = f

    file = os.path.join(path, file)
    energy = np.loadtxt(open(file,'rb'),delimiter=",",skiprows=1,usecols=[2])
    loss = np.loadtxt(open(file,'rb'),delimiter=",",skiprows=1,usecols=[3])
    y = energy                  
    z = np.log10(loss)
    x = range(len(energy))

    plt.figure()

    # 去除顶部和右边框框
    ax = plt.axes()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.xlabel('iters')    # x轴标签
    plt.ylabel('loss')     # y轴标签
	
    # 以x_train_loss为横坐标，y_train_loss为纵坐标，曲线宽度为1，实线，增加标签，训练损失，
    # 默认颜色，如果想更改颜色，可以增加参数color='red',这是红色。
    plt.plot(x, z, linewidth=1, linestyle="solid", label="train loss")
    plt.legend()
    plt.title('Loss curve')

    save_name = os.path.join(path, name + '_loss.png')
    plt.savefig(save_name)
    plt.close()

    plt.figure()

    # 去除顶部和右边框框
    ax = plt.axes()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.xlabel('iters')    # x轴标签
    plt.ylabel('energy')     # y轴标签
	
    # 以x_train_loss为横坐标，y_train_loss为纵坐标，曲线宽度为1，实线，增加标签，训练损失，
    # 默认颜色，如果想更改颜色，可以增加参数color='red',这是红色。
    plt.plot(x, y, linewidth=1, linestyle="solid", label="energy")
    plt.legend()
    plt.title('Energy curve')

    save_name = os.path.join(path, name + '_energy.png')
    plt.savefig(save_name)
    plt.close()    
    
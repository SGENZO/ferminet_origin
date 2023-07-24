"""用于绘制波函数的分布图像"""

import os
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import ticker, cm
import numpy as np
from jax import numpy as jnp
from typing import Optional
from ferminet import constants
from ferminet import networks

def save_plot(path: Optional[str],
              name: Optional[str],
              params: networks.ParamTree,
              batch_network: networks.LogFermiNetLike):

    xx = np.arange(-10, 10, 0.2)
    yy = np.arange(-10, 10, 0.2)
    X, Y = np.meshgrid(xx, yy)
    m,n = X.shape
    Xi = X.reshape(m*n, 1)
    Yi = Y.reshape(m*n, 1)
    Z = np.zeros(m*n)
    Zi = Z.reshape(m*n, 1)
    Input = np.concatenate((Xi,Yi,Zi), axis=1)
    Input = Input.reshape(1, m*n, 3)
    F = constants.pmap(batch_network)(params, Input)
    F = (jnp.exp(F)) ** 2
    F = F.reshape(m, n)

    fig2 = plt.figure()  #定义新的三维坐标轴
    ax2 = plt.axes(projection='3d')

    #作图
    ax2.plot_surface(X,Y,F,cmap='rainbow')
      #ax2.contour(X,Y,Z, zdim='z',offset=-2，cmap=plt.get_cmap('Spectral'))   #等高线图，要设置offset，为Z的最小值

    save_name = os.path.join(path, name + '.png')
    plt.savefig(save_name)
    # plt.show()


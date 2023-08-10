# 提供跟初值有关的loss function，传给train里面训练初值的optimizer

from typing import Tuple

import chex
from ferminet import constants
from ferminet import networks
import jax
import jax.numpy as jnp
from typing_extensions import Protocol
from ferminet.ibc import first_excited_state

@chex.dataclass
class AuxiliaryInitLossData:
    variance: jnp.DeviceArray
    squared_loss: jnp.DeviceArray


class InitLossFn(Protocol):

    def __call__(
        self,
        params: networks.ParamTree,
        key: chex.PRNGKey,
        data: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, AuxiliaryInitLossData]:
        """生成对一个batch的initial loss的函数类
    
        Args:
        params_previous: 
        key:
        data_previous:
        """
    
    
def make_loss(network: networks.LogFermiNetLike,
              initial_boundary: first_excited_state.InitialState) -> InitLossFn:
    
    batch_network = jax.vmap(network, in_axes=(None, 0), out_axes=0)
    batch_initial_boundary = jax.vmap(initial_boundary, in_axes=0, out_axes=0)
    
    def loss(
        params_previous: networks.ParamTree,
        key: chex.PRNGKey,
        data_previous: jnp.ndarray
    ) -> jnp.ndarray:
        
        f = batch_initial_boundary(data_previous)
        psi = batch_network(params_previous, data_previous)
        SE = (f - psi) ** 2 # f是分布函数，所以与psi^2做差 改成直接对psi监督
        MSE = constants.pmean(jnp.mean(SE))

        variance =constants.pmean(jnp.mean((SE - MSE)**2))
        
        return MSE, AuxiliaryInitLossData(variance=variance, squared_loss=SE)
    
    return loss

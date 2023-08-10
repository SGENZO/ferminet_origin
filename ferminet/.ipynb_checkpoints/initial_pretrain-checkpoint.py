from typing import Callable, Optional, Sequence, Tuple, Union

from absl import logging
import chex
from ferminet import constants
from ferminet import envelopes
from ferminet import mcmc
from ferminet import networks
from ferminet.utils import scf
from ferminet.utils import system
import jax
from jax import numpy as jnp
import kfac_jax
import numpy as np
import optax
from jax.scipy.special import sph_harm
from scipy.special import assoc_laguerre


def make_pretrain_step(batch_network: networks.LogFermiNetLike,
                       optimizer_update: optax.TransformUpdateFn,
                       full_det: bool = False):

  def initial_orbital(data:jnp.ndarray, n:jnp.int_, l:jnp.int_, m:jnp.int_):
          data = jnp.reshape(data, [-1,3])
          x = data[0]
          y = data[1]
          z = data[2]
          r = (x**2 + y**2 + z**2)**0.5
          theta = jnp.arccos(z/r)
          phi = jnp.arctan(y/x)
          theta = jnp.array(theta)
          phi = jnp.array(phi)
          rho = r / n
          Lag = assoc_laguerre(2 * rho, n - l - 1, 2 * l + 1)
          Sph = sph_harm(m, l, theta, phi)


          result = jnp.array(jnp.exp(-rho) * jnp.power((2*rho),l) * Lag * Sph)
          return result

                           
  def pretrain_step(data, params, state, key, logprob):
    """One iteration of pretraining to match HF."""

    def loss_fn(p, x):
        psi = batch_network(p, x)
        dshape = psi.shape
        f = initial_orbital(x,2,0,0)
        f = jnp.reshape(f, dshape)
        SE = (f - psi) ** 2 
        MSE = jnp.mean(SE) 
        return MSE#constants.pmean(MSE)

    val_and_grad = jax.value_and_grad(loss_fn, argnums=0)
    loss_val, search_direction = val_and_grad(params, data)
    #search_direction = constants.pmean(search_direction)
    updates, state = optimizer_update(search_direction, state, params)
    params = optax.apply_updates(params, updates)
    data, key, logprob, _ = mcmc.mh_update(params, batch_network, data, key,
                                           logprob, 0)
    return data, params, state, loss_val, logprob

  return pretrain_step


def pretrain_is(
    *,
    params: networks.ParamTree,
    data: jnp.ndarray,
    batch_network: networks.FermiNetLike,
    sharded_key: chex.PRNGKey,
    iterations: int = 1000,
    logger: Optional[Callable[[int, float], None]] = None,
):

  optimizer = optax.adam(3.e-4)
  opt_state_pt = constants.pmap(optimizer.init)(params)

  pretrain_step = make_pretrain_step(batch_network, optimizer.update)
  pretrain_step = constants.pmap(pretrain_step)
  pnetwork = constants.pmap(batch_network)
  logprob = 2. * pnetwork(params, data)

  for t in range(iterations):
    sharded_key, subkeys = kfac_jax.utils.p_split(sharded_key)
    data, params, opt_state_pt, loss, logprob = pretrain_step(
        data, params, opt_state_pt, subkeys, logprob)
    logging.info('Pretrain iter %05d: %g', t, loss[0])
    if logger:
      logger(t, loss[0])
  return params, data
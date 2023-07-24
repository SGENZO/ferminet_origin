# Copyright 2020 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Helper functions to create the loss and custom gradient of the loss."""

from typing import Tuple

import chex
from ferminet import constants
from ferminet import hamiltonian
from ferminet import networks
import jax
import scipy
import jax.numpy as jnp
import kfac_jax
from typing_extensions import Protocol


@chex.dataclass
class AuxiliaryLossData:
  """Auxiliary data returned by total_energy.

  Attributes:
    variance: mean variance over batch, and over all devices if inside a pmap.
    local_energy: local energy for each MCMC configuration.
  """
  variance: jnp.DeviceArray
  local_energy: jnp.DeviceArray
  local_for_psi: jnp.DeviceArray
  local_for_psi_cross: jnp.DeviceArray
  local_for_phi: jnp.DeviceArray
  local_for_phi_cross: jnp.DeviceArray
  local_pre_psi: jnp.DeviceArray
  local_pre_phi: jnp.DeviceArray


class LossFn(Protocol):

  def __call__(
      self,
      params_psi: networks.ParamTree,
      params_phi: networks.ParamTree,
      params_previous: networks.ParamTree,
      key: chex.PRNGKey,
      data_psi: jnp.ndarray,
      data_phi: jnp.ndarray,
  ) -> Tuple[jnp.ndarray, AuxiliaryLossData]:
    """Evaluates the total energy of the network for a batch of configurations.

    Note: the signature of this function is fixed to match that expected by
    kfac_jax.optimizer.Optimizer with value_func_has_rng=True and
    value_func_has_aux=True.

    Args:
      params: parameters to pass to the network.
      key: PRNG state.
      data: Batched electron positions to pass to the network.

    Returns:
      (loss, aux_data), where loss is the mean energy, and aux_data is an
      AuxiliaryLossData object containing the variance of the energy and the
      local energy per MCMC configuration. The loss and variance are averaged
      over the batch and over all devices inside a pmap.
    """


def make_loss(network: networks.LogFermiNetLike,
              local_energy: hamiltonian.LocalEnergy,
              clip_local_energy: float = 0.0) -> LossFn:
  """Creates the loss function, including custom gradients.

  Args:
    network: callable which evaluates the log of the magnitude of the
      wavefunction (square root of the log probability distribution) at a
      single MCMC configuration given the network parameters.
    local_energy: callable which evaluates the local energy.
    clip_local_energy: If greater than zero, clip local energies that are
      outside [E_L - n D, E_L + n D], where E_L is the mean local energy, n is
      this value and D the mean absolute deviation of the local energies from
      the mean, to the boundaries. The clipped local energies are only used to
      evaluate gradients.

  Returns:
    Callable with signature (params, data) and returns (loss, aux_data), where
    loss is the mean energy, and aux_data is an AuxiliaryLossDataobject. The
    loss is averaged over the batch and over all devices inside a pmap.
  """
  batch_local_energy = jax.vmap(local_energy, in_axes=(None, 0, 0), out_axes=0)
  batch_network = jax.vmap(network, in_axes=(None, 0), out_axes=0)

  # h = T/N 时间步长
  h_origin = 0.1
  # hbar = scipy.constants.hbar
  hbar = 1
  h = h_origin/hbar

  @jax.custom_jvp
  def total_energy(
      params_psi: networks.ParamTree,
      params_phi: networks.ParamTree,
      params_previous: networks.ParamTree,
      key: chex.PRNGKey,
      data_psi: jnp.ndarray,
      data_phi: jnp.ndarray,
  ) -> Tuple[jnp.ndarray, AuxiliaryLossData]:
    """Evaluates the total energy of the network for a batch of configurations.

    Note: the signature of this function is fixed to match that expected by
    kfac_jax.optimizer.Optimizer with value_func_has_rng=True and
    value_func_has_aux=True.

    Args:
      params_previous:
      params: parameters to pass to the network.
      key: PRNG state.
      data: Batched MCMC configurations to pass to the local energy function.

    Returns:
      (loss, aux_data), where loss is the mean energy, and aux_data is an
      AuxiliaryLossData object containing the variance of the energy and the
      local energy per MCMC configuration. The loss and variance are averaged
      over the batch and over all devices inside a pmap.
    """
    keys_psi = jax.random.split(key, num=data_psi.shape[0])
    keys_phi = jax.random.split(key, num=data_phi.shape[0])

    psi = batch_network(params_psi, data_psi)
    psi_cross = batch_network(params_psi, data_phi)
    phi = batch_network(params_phi, data_phi)
    phi_cross = batch_network(params_phi, data_psi)
    previous_psi = batch_network(params_previous, data_psi)
    previous_phi = batch_network(params_previous, data_phi)
    e_l_pre_psi = batch_local_energy(params_previous, keys_psi, data_psi)
    e_l_pre_phi = batch_local_energy(params_previous, keys_phi, data_phi)
      
    u_psi = -(h/2 * e_l_pre_psi * jnp.exp(previous_psi) + 1j * jnp.exp(previous_psi))
    u_phi = -(h/2 * e_l_pre_phi * jnp.exp(previous_phi) + 1j * jnp.exp(previous_phi))
    u_psi = jnp.log(u_psi)
    u_phi = jnp.log(u_phi)

    e_l_psi = batch_local_energy(params_psi, keys_psi, data_psi)
    e_l_psi_cross = batch_local_energy(params_psi, keys_psi, data_phi)
    e_l_phi = batch_local_energy(params_phi, keys_phi, data_phi)
    e_l_phi_cross = batch_local_energy(params_phi, keys_phi, data_psi)

    # 已经加入1j
    part_1 = h/2 * jnp.exp(psi_cross - phi) * e_l_psi_cross - \
             1j * jnp.exp(psi_cross - phi) - jnp.exp(u_phi - phi)
    part_2 = h/2 * jnp.exp(phi_cross - psi) * e_l_phi_cross + \
             1j * jnp.exp(phi_cross - psi) - jnp.exp(phi_cross + jnp.conj(u_psi) - psi - jnp.conj(psi))

    e_l = part_1 * part_2
    loss = constants.pmean(jnp.mean(e_l))
    loss = jnp.real(loss)

    # 注意这个variance是没有统计学意义的
    variance = constants.pmean(jnp.mean((e_l - loss)**2))
    variance = jnp.real(variance)
      
    return loss, AuxiliaryLossData(variance=variance, local_energy=e_l,
                                   local_for_psi=e_l_psi, local_for_psi_cross=e_l_psi_cross,
                                   local_for_phi=e_l_phi, local_for_phi_cross=e_l_phi_cross,
                                   local_pre_psi=e_l_pre_psi, local_pre_phi=e_l_pre_phi)

  @total_energy.defjvp
  def total_energy_jvp(primals, tangents):  # pylint: disable=unused-variable
    """Custom Jacobian-vector product for unbiased local energy gradients."""
    params_psi, params_phi, params_previous, key, data_psi, data_phi = primals
    loss, aux_data = total_energy(params_psi, params_phi, params_previous, key, data_psi, data_phi)

    if clip_local_energy > 0.0:
      # Try centering the window around the median instead of the mean?
      tv = jnp.mean(jnp.abs(aux_data.local_energy - loss))
      tv = constants.pmean(tv)
      diff = jnp.clip(aux_data.local_energy,
                      loss - clip_local_energy * tv,
                      loss + clip_local_energy * tv) - loss
    else:
      diff = aux_data.local_energy - loss

    e_l_psi = aux_data.local_for_psi
    e_l_psi_cross = aux_data.local_for_psi_cross
    e_l_phi = aux_data.local_for_phi
    e_l_phi_cross = aux_data.local_for_phi_cross

    psi = batch_network(params_psi, data_psi)
    psi_cross = batch_network(params_psi, data_phi)
    phi = batch_network(params_phi, data_phi)
    phi_cross = batch_network(params_phi, data_psi)
    previous_psi = batch_network(params_previous, data_psi)
    previous_phi = batch_network(params_previous, data_phi)
    e_l_pre_psi = aux_data.local_pre_psi
    e_l_pre_phi = aux_data.local_pre_phi
      
    u_psi = -(h/2 * e_l_pre_psi * jnp.exp(previous_psi) + 1j * jnp.exp(previous_psi))
    u_phi = -(h/2 * e_l_pre_phi * jnp.exp(previous_phi) + 1j * jnp.exp(previous_phi))
    u_psi = jnp.log(u_psi)
    u_phi = jnp.log(u_phi)

    part_1 = h/2 * jnp.exp(psi_cross - phi) * e_l_psi_cross - \
             1j * jnp.exp(psi_cross - phi) - jnp.exp(u_phi - phi)
    part_2 = h/2 * jnp.exp(phi_cross - psi) * e_l_phi_cross + \
             1j * jnp.exp(phi_cross - psi) - jnp.exp(phi_cross + jnp.conj(u_psi) - psi - jnp.conj(psi))

    # Due to the simultaneous requirements of KFAC (calling convention must be
    # (params, rng, data)) and Laplacian calculation (only want to take
    # Laplacian wrt electron positions) we need to change up the calling
    # convention between total_energy and batch_network
    primals_psi = primals[0], primals[4]
    tangents_psi = tangents[0], tangents[4]
    primals_phi = primals[1], primals[5]
    tangents_phi = tangents[1], tangents[5]

    primals_psi_cross = primals[0], primals[5]
    tangents_psi_cross = tangents[0], tangents[5]
    primals_phi_cross = primals[1], primals[4]
    tangents_phi_cross = tangents[1], tangents[4]

    psi_primal, psi_tangent = jax.jvp(batch_network, primals_psi, tangents_psi)
    psi_cross_primal, psi_cross_tangent = jax.jvp(batch_network, primals_psi_cross, tangents_psi_cross)
    phi_primal, phi_tangent = jax.jvp(batch_network, primals_phi, tangents_phi)
    phi_cross_primal, phi_cross_tangent = jax.jvp(batch_network, primals_phi_cross, tangents_phi_cross)

    # 这里广播是干啥的啊？上面这些向量怎么求均值？广播了这个分布是为了求均值吗？cross了咋办
    kfac_jax.register_normal_predictive_distribution(psi_primal[:, None])
    kfac_jax.register_normal_predictive_distribution(phi_primal[:, None])

    part_3 = jnp.conj(psi_tangent)
    part_4 = jnp.conj(phi_tangent)
    part_5 = jnp.conj(e_l_phi) * jnp.exp(psi_cross - phi) * psi_cross_tangent
    part_6 = jnp.conj(e_l_phi_cross) * jnp.conj(jnp.exp(phi_cross - psi)) * psi_tangent
    part_7 = jnp.exp(psi_cross - phi) * psi_cross_tangent
    part_8 = jnp.conj(jnp.exp(phi_cross - psi)) * psi_tangent
    part_9 = e_l_psi * jnp.conj(jnp.exp(phi_cross - psi) * phi_cross_tangent)
    part_10 = e_l_psi_cross * jnp.exp(psi_cross - phi) * jnp.conj(phi_tangent)
    part_11 = jnp.conj(jnp.exp(phi_cross - psi) * phi_cross_tangent)
    part_12 = jnp.exp(psi_cross - phi) * jnp.conj(phi_tangent)
    part_13 = jnp.exp(jnp.conj(phi_cross) + u_psi - jnp.conj(psi) - psi) * jnp.conj(phi_cross_tangent)
    part_14 = jnp.exp(u_phi - phi) * jnp.conj(phi_tangent)

    psi_gradient = (h/2 * part_5 - 1j * part_7) * part_2 + \
                   jnp.conj((h/2 * part_6 - 1j * part_8)) * part_1 - \
                   (2 * part_3) * part_1 * part_2
    phi_gradient = jnp.conj((h/2 * part_9 - 1j * part_11 - part_13)) * part_1 + \
                   (h/2 * part_10 - 1j * part_12 - part_14) * part_2 - \
                   (2 * part_4) * part_1 * part_2

    psi_gradient = jnp.mean(psi_gradient)
    phi_gradient = jnp.mean(phi_gradient)
    psi_gradient = jnp.real(psi_gradient)
    phi_gradient = jnp.real(phi_gradient)

    primals_out = loss, aux_data
    device_batch_size = jnp.shape(aux_data.local_energy)[0]
    # tangents_out = (jnp.dot(psi_tangent, diff) / device_batch_size, aux_data)
    tangents_out = ((psi_gradient + phi_gradient) / device_batch_size, aux_data)

    return primals_out, tangents_out

  return total_energy

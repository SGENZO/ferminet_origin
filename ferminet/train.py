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

"""Core training loop for neural QMC in JAX."""

import functools
import importlib
import time
from typing import Optional, Sequence, Tuple, Union

from absl import logging
import chex
from ferminet import checkpoint
from ferminet import constants
from ferminet import curvature_tags_and_blocks
from ferminet import envelopes
from ferminet import hamiltonian
from ferminet import loss as qmc_loss_functions
from ferminet import mcmc
from ferminet import networks
from ferminet import pretrain
from ferminet.utils import multi_host
from ferminet.utils import statistics
from ferminet.utils import system
from ferminet.utils import writers
import jax
import jax.numpy as jnp
import kfac_jax
import ml_collections
import numpy as np
import optax
from typing_extensions import Protocol


def init_electrons(
        key,
        molecule: Sequence[system.Atom],
        electrons: Sequence[int],
        batch_size: int,
        init_width: float,
) -> jnp.ndarray:
    """Initializes electron positions around each atom.

  Args:
    key: JAX RNG state.
    molecule: system.Atom objects making up the molecule.
    electrons: tuple of number of alpha and beta electrons.
    batch_size: total number of MCMC configurations to generate across all
      devices.
    init_width: width of (atom-centred) Gaussian used to generate initial
      electron configurations.

  Returns:
    array of (batch_size, (nalpha+nbeta)*ndim) of initial (random) electron
    positions in the initial MCMC configurations and ndim is the dimensionality
    of the space (i.e. typically 3).
  """
    if sum(atom.charge for atom in molecule) != sum(electrons):
        if len(molecule) == 1:
            atomic_spin_configs = [electrons]
        else:
            raise NotImplementedError('No initialization policy yet '
                                      'exists for charged molecules.')
    else:
        atomic_spin_configs = [
            (atom.element.nalpha, atom.element.nbeta) for atom in molecule
        ]
        assert sum(sum(x) for x in atomic_spin_configs) == sum(electrons)
        while tuple(sum(x) for x in zip(*atomic_spin_configs)) != electrons:
            i = np.random.randint(len(atomic_spin_configs))
            nalpha, nbeta = atomic_spin_configs[i]
            atomic_spin_configs[i] = nbeta, nalpha

    # Assign each electron to an atom initially.
    electron_positions = []
    for i in range(2):
        for j in range(len(molecule)):
            atom_position = jnp.asarray(molecule[j].coords)
            electron_positions.append(
                jnp.tile(atom_position, atomic_spin_configs[j][i]))
    electron_positions = jnp.concatenate(electron_positions)
    # Create a batch of configurations with a Gaussian distribution about each
    # atom.
    key, subkey = jax.random.split(key)
    return (
            electron_positions +
            init_width *
            jax.random.normal(subkey, shape=(batch_size, electron_positions.size)))


# All optimizer states (KFAC and optax-based).  OptState是否需要用两个？后面Adam的init只能传入一个params
OptimizerState = Union[optax.OptState, kfac_jax.optimizer.OptimizerState]
OptUpdateResults = Tuple[networks.ParamTree, networks.ParamTree,  # param_psi, param_phi
                         Optional[OptimizerState], Optional[OptimizerState],  # OptState
                         jnp.ndarray,  # loss
                         Optional[qmc_loss_functions.AuxiliaryLossData]]


class OptUpdate(Protocol):

    def __call__(self,
                 params_psi: networks.ParamTree, data_psi: jnp.ndarray,
                 params_phi: networks.ParamTree, data_phi: jnp.ndarray,
                 params_previous: networks.ParamTree,
                 opt_state_psi: optax.OptState, opt_state_phi: optax.OptState,
                 key: chex.PRNGKey) -> OptUpdateResults:
        """Evaluates the loss and gradients and updates the parameters accordingly.

    Args:
      params_psi: network parameters.
      params_phi: network parameters.
      params_previous: network parameters.
      data_psi: electron positions.
      data_phi: electron positions.
      opt_state: optimizer internal state.
      key: RNG state.

    Returns:
      Tuple of (params_psi, params_phi, opt_state, loss, aux_data), where params and opt_state
      are the updated parameters and optimizer state, loss is the evaluated loss
      and aux_data auxiliary data (see AuxiliaryLossData docstring).
    """


StepResults = Tuple[jnp.ndarray, jnp.ndarray,  # data_psi 和 data_phi
                    networks.ParamTree, networks.ParamTree,  # params_psi 和 params_phi
                    Optional[optax.OptState],  Optional[optax.OptState],  # OptState
                    jnp.ndarray, qmc_loss_functions.AuxiliaryLossData,  # loss, aux_data
                    jnp.ndarray, jnp.ndarray]  # pmove_psi, pmove_phi


class Step(Protocol):

    def __call__(self,
                 data_psi: jnp.ndarray,
                 data_phi: jnp.ndarray,
                 params_psi: networks.ParamTree,
                 params_phi: networks.ParamTree,
                 params_previous: networks.ParamTree,
                 state_psi: OptimizerState,
                 state_phi: OptimizerState,
                 key: chex.PRNGKey,
                 mcmc_width_psi: jnp.ndarray,
                 mcmc_width_phi: jnp.ndarray) -> StepResults:
        """Performs one set of MCMC moves and an optimization step.

    Args:
      data: batch of MCMC configurations.
      params: network parameters.
      state: optimizer internal state.
      key: JAX RNG state.
      mcmc_width: width of MCMC move proposal. See mcmc.make_mcmc_step.

    Returns:
      Tuple of (data, params, state, loss, aux_data, pmove).
        data: Updated MCMC configurations drawn from the network given the
          *input* network parameters.
        params: updated network parameters after the gradient update.
        state: updated optimization state.
        loss: energy of system based on input network parameters averaged over
          the entire set of MCMC configurations.
        aux_data: AuxiliaryLossData object also returned from evaluating the
          loss of the system.
        pmove: probability that a proposed MCMC move was accepted.
    """


def null_update(params_psi: networks.ParamTree, data_psi: jnp.ndarray,
                params_phi: networks.ParamTree, data_phi: jnp.ndarray,
                params_previous: networks.ParamTree,
                opt_state_psi: Optional[optax.OptState],
                opt_state_phi: Optional[optax.OptState],
                key: chex.PRNGKey) -> OptUpdateResults:
    """Performs an identity operation with an OptUpdate interface."""
    del data_psi, data_phi, key
    return params_psi, params_phi, opt_state_psi, opt_state_phi, jnp.zeros(1), None


# 这里用closure方法，改成了先对psi下降一步，再对phi上升一步
def make_opt_update_step(evaluate_loss: qmc_loss_functions.LossFn,
                         optimizer_psi: optax.GradientTransformation,
                         optimizer_phi: optax.GradientTransformation,
                         iteration_psi: int, iteration_phi: int) -> OptUpdate:
    """Returns an OptUpdate function for performing a parameter update."""

    # Differentiate wrt parameters (argument 0)
    # loss_and_grad_psi = jax.value_and_grad(evaluate_loss, argnums=0, has_aux=True)
    # loss_and_grad_phi = jax.value_and_grad(evaluate_loss, argnums=1, has_aux=True)

    def opt_update(params_psi: networks.ParamTree, data_psi: jnp.ndarray,
                   params_phi: networks.ParamTree, data_phi: jnp.ndarray,
                   params_previous: networks.ParamTree,
                   opt_state_psi: Optional[optax.OptState],
                   opt_state_phi: Optional[optax.OptState],
                   key: chex.PRNGKey) -> OptUpdateResults:
        """Evaluates the loss and gradients and updates the parameters using optax."""
        # 对loss进行closure操作
        # 先psi下降一步
        evaluate_loss_psi = lambda params, keys, data: \
            evaluate_loss(params, params_phi, params_previous, keys, data, data_phi)
        loss_and_grad_psi = jax.value_and_grad(evaluate_loss_psi, argnums=0, has_aux=True)

        for k in range(iteration_psi):
            (loss, aux_data), grad_psi = loss_and_grad_psi(params_psi, key, data_psi)
            grad_psi = constants.pmean(grad_psi)
            updates_psi, opt_state_psi = optimizer_psi.update(grad_psi, opt_state_psi, params_psi)
            params_psi = optax.apply_updates(params_psi, updates_psi)

        # 再对phi上升一步 这里有个负号
        evaluate_loss_phi = lambda params, keys, data: \
            evaluate_loss(params_psi, params, params_previous, keys, data_psi, data)
        loss_and_grad_phi = jax.value_and_grad(evaluate_loss_phi, argnums=0, has_aux=True)

        for k in range(iteration_phi):
            (loss, aux_data), grad_phi = loss_and_grad_phi(params_phi, key, data_phi)
            grad_phi = constants.pmean(grad_phi)
            updates_phi, opt_state_phi = optimizer_phi.update(grad_phi, opt_state_phi, params_phi)
            params_phi = optax.apply_updates(params_phi, updates_phi)

        # 原先使用同一个优化器，所以要对updates乘以-1.updates为dict{list,list,dict,list,list}.
        # 现在改成两个优化器，在optax里面scale调整正负1就可以了
        # for key in updates_phi:
        # if type(key) == dict:
        # for subkey in updates_phi[key]:
        # updates_phi[key][subkey] = -1 * updates_phi[key][subkey]
        # if type(key) == list:
        # updates_phi[key] = -1 * updates_phi[key]

        return params_psi, params_phi, opt_state_psi, opt_state_phi, loss, aux_data

    return opt_update


def make_loss_step(evaluate_loss: qmc_loss_functions.LossFn) -> OptUpdate:
    """Returns an OptUpdate function for evaluating the loss."""

    def loss_eval(params_psi: networks.ParamTree, data_psi: jnp.ndarray,
                  params_phi: networks.ParamTree, data_phi: jnp.ndarray,
                  params_previous: networks.ParamTree,
                  opt_state_psi: Optional[optax.OptState],
                  opt_state_phi: Optional[optax.OptState],
                  key: chex.PRNGKey) -> OptUpdateResults:
        """Evaluates just the loss and gradients with an OptUpdate interface."""
        loss, aux_data = evaluate_loss(params_psi, params_phi, params_previous, key, data_psi, data_phi)

        return params_psi, params_phi, opt_state_psi, opt_state_phi, loss, aux_data

    return loss_eval


def make_training_step(
        mcmc_step,
        optimizer_step: OptUpdate,
) -> Step:
    """Factory to create traning step for non-KFAC optimizers.

  Args:
    mcmc_step: Callable which performs the set of MCMC steps. See make_mcmc_step
      for creating the callable.
    optimizer_step: OptUpdate callable which evaluates the forward and backward
      passes and updates the parameters and optimizer state, as required.

  Returns:
    step, a callable which performs a set of MCMC steps and then an optimization
    update. See the Step protocol for details.
  """

    # 这个修饰是干啥的
    #@functools.partial(constants.pmap, donate_argnums=(0, 1, 2, 3, 4, 5, 6))
    @constants.pmap
    def step(data_psi: jnp.ndarray, data_phi: jnp.ndarray,
             params_psi: networks.ParamTree, params_phi: networks.ParamTree, params_previous: networks.ParamTree,
             state_psi: Optional[optax.OptState],
             state_phi: Optional[optax.OptState],
             key: chex.PRNGKey,
             mcmc_width_psi: jnp.ndarray,
             mcmc_width_phi: jnp.ndarray) -> StepResults:
        """A full update iteration (except for KFAC): MCMC steps + optimization."""
        # MCMC loop for psi
        mcmc_key, loss_key = jax.random.split(key, num=2)
        data_psi, pmove_psi = mcmc_step(params_psi, data_psi, mcmc_key, mcmc_width_psi)
        # MCMC loop for phi
        mcmc_key, loss_key = jax.random.split(key, num=2)
        data_phi, pmove_phi = mcmc_step(params_phi, data_phi, mcmc_key, mcmc_width_phi)

        # Optimization step for psi&phi 需要提前在optimizer里调整好内循环的次数
        new_params_psi, new_params_phi, state_psi, state_phi, loss, aux_data = \
            optimizer_step(params_psi, data_psi, params_phi, data_phi, params_previous, state_psi, state_phi, loss_key)

        result: Tuple
        result = [data_psi, data_phi,
                  new_params_psi, new_params_phi,
                  state_psi, state_phi,
                  loss, aux_data, pmove_psi, pmove_phi]
        return result

    return step


def make_kfac_training_step(mcmc_step, damping: float,
                            optimizer: kfac_jax.Optimizer) -> Step:
    """Factory to create training step for KFAC optimizers.

  Args:
    mcmc_step: Callable which performs the set of MCMC steps. See make_mcmc_step
      for creating the callable.
    damping: value of damping to use for each KFAC update step.
    optimizer: KFAC optimizer instance.

  Returns:
    step, a callable which performs a set of MCMC steps and then an optimization
    update. See the Step protocol for details.
  """
    mcmc_step = constants.pmap(mcmc_step, donate_argnums=1)
    shared_mom = kfac_jax.utils.replicate_all_local_devices(jnp.zeros([]))
    shared_damping = kfac_jax.utils.replicate_all_local_devices(
        jnp.asarray(damping))

    def step(data_psi: jnp.ndarray, data_phi: jnp.ndarray,
             params_psi: networks.ParamTree, params_phi: networks.ParamTree, params_previous: networks.ParamTree,
             state: kfac_jax.optimizer.OptimizerState,
             key: chex.PRNGKey, mcmc_width: jnp.ndarray) -> StepResults:
        """A full update iteration for KFAC: MCMC steps + optimization."""
        # KFAC requires control of the loss and gradient eval, so everything called
        # here must be already pmapped.

        # MCMC loop for psi
        mcmc_keys, loss_keys = kfac_jax.utils.p_split(key)
        new_data_psi, pmove_psi = mcmc_step(params_psi, data_psi, mcmc_keys, mcmc_width)

        # Optimization step
        new_params_psi, state, stats = optimizer.step(
            params=params_psi,
            state=state,
            rng=loss_keys,
            data_iterator=iter([new_data_psi]),
            momentum=shared_mom,
            damping=shared_damping)

        # MCMC loop for phi
        mcmc_keys, loss_keys = kfac_jax.utils.p_split(key)
        new_data_phi, pmove_phi = mcmc_step(params_phi, data_phi, mcmc_keys, mcmc_width)

        # Optimization step
        new_params_phi, state, stats = optimizer.step(
            params=params_phi,
            state=state,
            rng=loss_keys,
            data_iterator=iter([new_data_phi]),
            momentum=shared_mom,
            damping=shared_damping)

        return new_data_psi, new_data_phi, new_params_psi, new_params_phi, \
               state, stats['loss'], stats['aux'], pmove_psi, pmove_psi

    return step


def train(cfg: ml_collections.ConfigDict, writer_manager=None):
    """Runs training loop for QMC.

  Args:
    cfg: ConfigDict containing the system and training parameters to run on. See
      base_config.default for more details.
    writer_manager: context manager with a write method for logging output. If
      None, a default writer (ferminet.utils.writers.Writer) is used.

  Raises:
    ValueError: if an illegal or unsupported value in cfg is detected.
  """
    # Device logging
    num_devices = jax.local_device_count()
    num_hosts = jax.device_count() // num_devices
    logging.info('Starting QMC with %i XLA devices per host '
                 'across %i hosts.', num_devices, num_hosts)
    if cfg.batch_size % (num_devices * num_hosts) != 0:
        raise ValueError('Batch size must be divisible by number of devices, '
                         f'got batch size {cfg.batch_size} for '
                         f'{num_devices * num_hosts} devices.')
    host_batch_size = cfg.batch_size // num_hosts  # batch size per host
    device_batch_size = host_batch_size // num_devices  # batch size per device
    data_shape = (num_devices, device_batch_size)

    # Check if mol is a pyscf molecule and convert to internal representation
    if cfg.system.pyscf_mol:
        cfg.update(
            system.pyscf_mol_to_internal_representation(cfg.system.pyscf_mol))

    # Convert mol config into array of atomic positions and charges
    atoms = jnp.stack([jnp.array(atom.coords) for atom in cfg.system.molecule])
    charges = jnp.array([atom.charge for atom in cfg.system.molecule])
    nspins = cfg.system.electrons

    if cfg.debug.deterministic:
        seed = 23
    else:
        seed = 1e6 * time.time()
        seed = int(multi_host.broadcast_to_hosts(seed))
    key = jax.random.PRNGKey(seed)

    # Create parameters, network, and vmaped/pmaped derivations

    if cfg.pretrain.method == 'direct_init' or (
            cfg.pretrain.method == 'hf' and cfg.pretrain.iterations > 0):
        hartree_fock = pretrain.get_hf(
            pyscf_mol=cfg.system.get('pyscf_mol'),
            molecule=cfg.system.molecule,
            nspins=nspins,
            restricted=False,
            basis=cfg.pretrain.basis)
        # broadcast the result of PySCF from host 0 to all other hosts
        hartree_fock.mean_field.mo_coeff = tuple([
            multi_host.broadcast_to_hosts(x)
            for x in hartree_fock.mean_field.mo_coeff
        ])

    hf_solution = hartree_fock if cfg.pretrain.method == 'direct_init' else None

    if cfg.network.make_feature_layer_fn:
        feature_layer_module, feature_layer_fn = (
            cfg.network.make_feature_layer_fn.rsplit('.', maxsplit=1))
        feature_layer_module = importlib.import_module(feature_layer_module)
        make_feature_layer = getattr(feature_layer_module, feature_layer_fn)
        feature_layer = make_feature_layer(
            charges,
            cfg.system.electrons,
            cfg.system.ndim,
            **cfg.network.make_feature_layer_kwargs)  # type: networks.FeatureLayer
    else:
        feature_layer = networks.make_ferminet_features(
            charges,
            cfg.system.electrons,
            cfg.system.ndim,
        )

    if cfg.network.make_envelope_fn:
        envelope_module, envelope_fn = (
            cfg.network.make_envelope_fn.rsplit('.', maxsplit=1))
        envelope_module = importlib.import_module(envelope_module)
        make_envelope = getattr(envelope_module, envelope_fn)
        envelope = make_envelope(**cfg.network.make_envelope_kwargs)  # type: envelopes.Envelope
    else:
        envelope = envelopes.make_isotropic_envelope()

    network_init, signed_network, network_options = networks.make_fermi_net(
        atoms,
        nspins,
        charges,
        envelope=envelope,
        feature_layer=feature_layer,
        bias_orbitals=cfg.network.bias_orbitals,
        use_last_layer=cfg.network.use_last_layer,
        hf_solution=hf_solution,
        full_det=cfg.network.full_det,
        ndim=cfg.system.ndim,
        **cfg.network.detnet)
    key, subkey = jax.random.split(key)
    params_psi = network_init(subkey)
    params_psi = kfac_jax.utils.replicate_all_local_devices(params_psi)
    key, subkey = jax.random.split(key)
    params_phi = network_init(subkey)
    params_phi = kfac_jax.utils.replicate_all_local_devices(params_phi)
    # Often just need log|psi(x)|.
    network = lambda *args, **kwargs: signed_network(*args, **kwargs)[1]  # type: networks.LogFermiNetLike
    batch_network = jax.vmap(
        network, in_axes=(None, 0), out_axes=0)  # batched network

    # Set up checkpointing and restore params/data if necessary
    # Mirror behaviour of checkpoints in TF FermiNet.
    # Checkpoints are saved to save_path.
    # When restoring, we first check for a checkpoint in save_path. If none are
    # found, then we check in restore_path.  This enables calculations to be
    # started from a previous calculation but then resume from their own
    # checkpoints in the event of pre-emption.

    ckpt_save_path_psi = checkpoint.create_save_path(cfg.log.save_path_psi)
    ckpt_restore_path_psi = checkpoint.get_restore_path(cfg.log.restore_path_psi)
    ckpt_save_path_phi = checkpoint.create_save_path(cfg.log.save_path_phi)
    ckpt_restore_path_phi = checkpoint.get_restore_path(cfg.log.restore_path_phi)

    ckpt_restore_filename_psi = (
            checkpoint.find_last_checkpoint(ckpt_save_path_psi) or
            checkpoint.find_last_checkpoint(ckpt_restore_path_psi))
    ckpt_restore_filename_phi = (
            checkpoint.find_last_checkpoint(ckpt_save_path_phi) or
            checkpoint.find_last_checkpoint(ckpt_restore_path_phi))

    # psi数据读取or生成
    if ckpt_restore_filename_psi:
        t_init, data_psi, params_psi, opt_state_ckpt_psi, mcmc_width_ckpt_psi = checkpoint.restore(
            ckpt_restore_filename_psi, host_batch_size)
    else:
        logging.info('No checkpoint of psi found. Training new model for psi.')
        key, subkey = jax.random.split(key)
        # make sure data on each host is initialized differently
        subkey = jax.random.fold_in(subkey, jax.process_index())
        data_psi = init_electrons(
            subkey,
            cfg.system.molecule,
            cfg.system.electrons,
            batch_size=host_batch_size,
            init_width=cfg.mcmc.init_width)
        data_psi = jnp.reshape(data_psi, data_shape + data_psi.shape[1:])
        data_psi = kfac_jax.utils.broadcast_all_local_devices(data_psi)
        t_init = 0
        opt_state_ckpt_psi = None
        mcmc_width_ckpt_psi = None

    # phi数据读取or生成
    if ckpt_restore_filename_phi:
        t_init, data_phi, params_phi, opt_state_ckpt_phi, mcmc_width_ckpt_phi = checkpoint.restore(
            ckpt_restore_filename_phi, host_batch_size)
    else:
        logging.info('No checkpoint of phi found. Training new model for phi.')
        key, subkey = jax.random.split(key)
        # make sure data on each host is initialized differently
        subkey = jax.random.fold_in(subkey, jax.process_index())
        data_phi = init_electrons(
            subkey,
            cfg.system.molecule,
            cfg.system.electrons,
            batch_size=host_batch_size,
            init_width=cfg.mcmc.init_width)
        data_phi = jnp.reshape(data_phi, data_shape + data_phi.shape[1:])
        data_phi = kfac_jax.utils.broadcast_all_local_devices(data_phi)
        t_init = 0
        opt_state_ckpt_phi = None
        mcmc_width_ckpt_phi = None

    # Set up logging 这里对pmove进行了psi和phi的区分
    train_schema = ['step', 'energy', 'ewmean', 'ewvar', 'pmove_psi', 'pmove_phi']

    # Initialisation done. We now want to have different PRNG streams on each
    # device. Shard the key over devices
    sharded_key = kfac_jax.utils.make_different_rng_key_on_all_devices(key)

    # Pretraining to match Hartree-Fock

    if (t_init == 0 and cfg.pretrain.method == 'hf' and
            cfg.pretrain.iterations > 0):
        orbitals = functools.partial(
            networks.fermi_net_orbitals,
            atoms=atoms,
            nspins=cfg.system.electrons,
            options=network_options,
        )
        batch_orbitals = jax.vmap(
            lambda params, data: orbitals(params, data)[0],
            in_axes=(None, 0),
            out_axes=0)
        sharded_key, subkeys = kfac_jax.utils.p_split(sharded_key)
        params_psi, data_psi = pretrain.pretrain_hartree_fock(
            params=params_psi,
            data=data_psi,
            batch_network=batch_network,
            batch_orbitals=batch_orbitals,
            network_options=network_options,
            sharded_key=subkeys,
            atoms=atoms,
            electrons=cfg.system.electrons,
            scf_approx=hartree_fock,
            iterations=cfg.pretrain.iterations)
        sharded_key, subkeys = kfac_jax.utils.p_split(sharded_key)
        params_phi, data_phi = pretrain.pretrain_hartree_fock(
            params=params_phi,
            data=data_phi,
            batch_network=batch_network,
            batch_orbitals=batch_orbitals,
            network_options=network_options,
            sharded_key=subkeys,
            atoms=atoms,
            electrons=cfg.system.electrons,
            scf_approx=hartree_fock,
            iterations=cfg.pretrain.iterations)

    # Main training

    # Construct MCMC step
    atoms_to_mcmc = atoms if cfg.mcmc.scale_by_nuclear_distance else None
    mcmc_step = mcmc.make_mcmc_step(
        batch_network,
        device_batch_size,
        steps=cfg.mcmc.steps,
        atoms=atoms_to_mcmc,
        one_electron_moves=cfg.mcmc.one_electron,
    )
    # Construct loss and optimizer
    if cfg.system.make_local_energy_fn:
        local_energy_module, local_energy_fn = (
            cfg.system.make_local_energy_fn.rsplit('.', maxsplit=1))
        local_energy_module = importlib.import_module(local_energy_module)
        make_local_energy = getattr(local_energy_module, local_energy_fn)  # type: hamiltonian.MakeLocalEnergy
        local_energy = make_local_energy(
            f=signed_network,
            atoms=atoms,
            charges=charges,
            nspins=nspins,
            use_scan=False,
            **cfg.system.make_local_energy_kwargs)
    else:
        local_energy = hamiltonian.local_energy(
            f=signed_network,
            atoms=atoms,
            charges=charges,
            nspins=nspins,
            use_scan=False)
    evaluate_loss = qmc_loss_functions.make_loss(
        network,
        local_energy,
        clip_local_energy=cfg.optim.clip_el)

    # Compute the learning rate
    def learning_rate_schedule(t_: jnp.ndarray) -> jnp.ndarray:
        return cfg.optim.lr.rate * jnp.power(
            (1.0 / (1.0 + (t_ / cfg.optim.lr.delay))), cfg.optim.lr.decay)

    # Construct and setup optimizer
    if cfg.optim.optimizer == 'none':
        optimizer_psi = None
        optimizer_phi = None
    elif cfg.optim.optimizer == 'adam':
        # 对两个优化器可以定义不同的schedule
        optimizer_psi = optax.chain(
            optax.scale_by_adam(**cfg.optim.adam),
            optax.scale_by_schedule(learning_rate_schedule),
            optax.scale(-1.))
        optimizer_phi = optax.chain(
            optax.scale_by_adam(**cfg.optim.adam),
            optax.scale_by_schedule(learning_rate_schedule),
            optax.scale(1.))
    elif cfg.optim.optimizer == 'lamb':
        optimizer_psi = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.scale_by_adam(eps=1e-7),
            optax.scale_by_trust_ratio(),
            optax.scale_by_schedule(learning_rate_schedule),
            optax.scale(-1))
        optimizer_phi = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.scale_by_adam(eps=1e-7),
            optax.scale_by_trust_ratio(),
            optax.scale_by_schedule(learning_rate_schedule),
            optax.scale(1))
    elif cfg.optim.optimizer == 'kfac':
        # Differentiate wrt parameters (argument 0)
        val_and_grad_theta = jax.value_and_grad(evaluate_loss, argnums=0, has_aux=True)
        val_and_grad_eta = jax.value_and_grad(evaluate_loss, argnums=1, has_aux=True)
        val_and_grad = val_and_grad_theta + val_and_grad_eta
        optimizer = kfac_jax.Optimizer(
            val_and_grad,
            l2_reg=cfg.optim.kfac.l2_reg,
            norm_constraint=cfg.optim.kfac.norm_constraint,
            value_func_has_aux=True,
            value_func_has_rng=True,
            learning_rate_schedule=learning_rate_schedule,
            curvature_ema=cfg.optim.kfac.cov_ema_decay,
            inverse_update_period=cfg.optim.kfac.invert_every,
            min_damping=cfg.optim.kfac.min_damping,
            num_burnin_steps=0,
            register_only_generic=cfg.optim.kfac.register_only_generic,
            estimation_mode='fisher_exact',
            multi_device=True,
            pmap_axis_name=constants.PMAP_AXIS_NAME,
            auto_register_kwargs=dict(
                graph_patterns=curvature_tags_and_blocks.GRAPH_PATTERNS,
            ),
            # debug=True
        )
        sharded_key, subkeys = kfac_jax.utils.p_split(sharded_key)
        opt_state = optimizer.init(params_psi, subkeys, data_psi)
        opt_state = opt_state_ckpt_psi or opt_state  # avoid overwriting ckpted state
    else:
        raise ValueError(f'Not a recognized optimizer: {cfg.optim.optimizer}')

    # 定义内循环步数
    k_psi = cfg.optim.iterations_psi
    k_phi = cfg.optim.iterations_phi
    if not optimizer:
        opt_state_psi = None
        opt_state_phi = None
        step = make_training_step(
            mcmc_step=mcmc_step,
            optimizer_step=make_loss_step(evaluate_loss))
    elif isinstance(optimizer, optax.GradientTransformation):
        # optax/optax-compatible optimizer (ADAM, LAMB, ...)
        opt_state_psi = jax.pmap(optimizer_psi.init)(params_psi)
        opt_state_psi = opt_state_ckpt_psi or opt_state_psi  # avoid overwriting ckpted state
        opt_state_phi = jax.pmap(optimizer_phi.init)(params_phi)
        opt_state_phi = opt_state_ckpt_psi or opt_state_phi  # avoid overwriting ckpted state
        step = make_training_step(
            mcmc_step=mcmc_step,
            optimizer_step=make_opt_update_step(evaluate_loss, optimizer_psi, optimizer_phi, k_psi, k_phi))
    elif isinstance(optimizer, kfac_jax.Optimizer): # kfac还没改
        step = make_kfac_training_step(
            mcmc_step=mcmc_step,
            damping=cfg.optim.kfac.damping,
            optimizer=optimizer)
    else:
        raise ValueError(f'Unknown optimizer: {optimizer}')

    # psi的mcmc步长设置
    if mcmc_width_ckpt_psi is not None:
        mcmc_width_psi = kfac_jax.utils.replicate_all_local_devices(mcmc_width_ckpt_psi[0])
    else:
        mcmc_width_psi = kfac_jax.utils.replicate_all_local_devices(
            jnp.asarray(cfg.mcmc.move_width))
    # phi的mcmc步长设置
    if mcmc_width_ckpt_phi is not None:
        mcmc_width_phi = kfac_jax.utils.replicate_all_local_devices(mcmc_width_ckpt_phi[0])
    else:
        mcmc_width_phi = kfac_jax.utils.replicate_all_local_devices(
            jnp.asarray(cfg.mcmc.move_width))
    # pmoves是做什么的，需不需要改
    pmoves_psi = np.zeros(cfg.mcmc.adapt_frequency)
    pmoves_phi = np.zeros(cfg.mcmc.adapt_frequency)

    # 这里写一个判断，从ckpt中读取previous
    params_previous = params_psi

    if t_init == 0:
        logging.info('Burning in MCMC chain for %d steps', cfg.mcmc.burn_in)

        burn_in_step = make_training_step(
            mcmc_step=mcmc_step, optimizer_step=null_update)

        for t in range(cfg.mcmc.burn_in):
            sharded_key, subkeys = kfac_jax.utils.p_split(sharded_key)
            data_psi, data_phi, params_psi, params_phi, *_ = burn_in_step(
                data_psi,
                data_phi,
                params_psi,
                params_phi,
                params_previous,
                state_psi=None,
                state_phi=None,
                key=subkeys,
                mcmc_width_psi=mcmc_width_psi,
                mcmc_width_phi=mcmc_width_phi)
        logging.info('Completed burn-in MCMC steps')
        sharded_key, subkeys = kfac_jax.utils.p_split(sharded_key)
        ptotal_energy = constants.pmap(evaluate_loss)
        initial_energy, _ = ptotal_energy(params_psi, params_psi, params_previous, \
                                          subkeys, data_psi, data_phi)
        logging.info('Initial energy: %03.4f E_h', initial_energy[0])

    time_of_last_ckpt = time.time()
    weighted_stats = None

    if cfg.optim.optimizer == 'none' and opt_state_ckpt_psi is not None: #这里需要仔细看一下,还没改
        # If opt_state_ckpt is None, then we're restarting from a previous inference
        # run (most likely due to preemption) and so should continue from the last
        # iteration in the checkpoint. Otherwise, starting an inference run from a
        # training run.
        logging.info('No optimizer provided. Assuming inference run.')
        logging.info('Setting initial iteration to 0.')
        t_init = 0

    if writer_manager is None:
        writer_manager = writers.Writer(
            name='train_stats',
            schema=train_schema,
            directory=ckpt_save_path_psi,
            # 这里能存两个directory吗
            iteration_key=None,
            log=False)
    with writer_manager as writer:
        # Main training loop
        for t in range(t_init, cfg.optim.iterations):
            sharded_key, subkeys = kfac_jax.utils.p_split(sharded_key)
            data_psi, data_phi, params_psi, params_phi, \
            opt_state_psi, opt_state_phi, loss, unused_aux_data, \
            pmove_psi, pmove_phi = step(
                data_psi, data_phi,
                params_psi, params_phi, params_previous,
                opt_state_psi, opt_state_phi,
                subkeys,
                mcmc_width_psi,
                mcmc_width_phi)

            # due to pmean, loss, and pmove should be the same across
            # devices.
            loss = loss[0]
            # per batch variance isn't informative. Use weighted mean and variance
            # instead.
            weighted_stats = statistics.exponentialy_weighted_stats(
                alpha=0.1, observation=loss, previous_stats=weighted_stats)
            pmove_psi = pmove_psi[0]
            pmove_phi = pmove_phi[0]

            # Update MCMC move width for psi and phi
            if t > 0 and t % cfg.mcmc.adapt_frequency == 0:
                if np.mean(pmoves_psi) > 0.55:
                    mcmc_width_psi *= 1.1
                if np.mean(pmoves_psi) < 0.5:
                    mcmc_width_psi /= 1.1
                pmoves_psi[:] = 0
                if np.mean(pmoves_phi) > 0.55:
                    mcmc_width_phi *= 1.1
                if np.mean(pmoves_phi) < 0.5:
                    mcmc_width_phi /= 1.1
                pmoves_phi[:] = 0
            pmoves_psi[t % cfg.mcmc.adapt_frequency] = pmove_psi
            pmoves_phi[t % cfg.mcmc.adapt_frequency] = pmove_phi

            if cfg.debug.check_nan:
                tree = {'params_psi': params_psi, 'params_phi': params_phi, 'loss': loss}
                if cfg.optim.optimizer != 'none':
                    tree['optim'] = opt_state
                chex.assert_tree_all_finite(tree)

            # Logging
            if t % cfg.log.stats_frequency == 0:
                logging.info(
                    'Step %05d: %03.4f E_h, exp. variance=%03.4f E_h^2, pmove=%0.2f', t,
                    loss, weighted_stats.variance, pmove_psi, pmove_phi)
                writer.write(
                    t,
                    step=t,
                    energy=np.asarray(loss),
                    ewmean=np.asarray(weighted_stats.mean),
                    ewvar=np.asarray(weighted_stats.variance),
                    pmove_psi=np.asarray(pmove_psi),
                    pmove_phi=np.asarray(pmove_phi))

            # Checkpointing
            if time.time() - time_of_last_ckpt > cfg.log.save_frequency * 60:
                checkpoint.save(ckpt_save_path_psi, t, data_psi, params_psi, \
                                opt_state_psi, mcmc_width_psi)
                checkpoint.save(ckpt_save_path_phi, t, data_phi, params_phi, \
                                opt_state_phi, mcmc_width_phi)
                time_of_last_ckpt = time.time()

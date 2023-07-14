from typing import Any, Sequence

import chex
from ferminet import networks
import jax
from jax import lax
import jax.numpy as jnp
from typing_extensions import Protocol


class InitialState(Protocol):

    def __call__(self, data: jnp.ndarray) -> jnp.ndarray:
        """Returns the initial state funcion at a configuration.
    
        Args:
          data: MCMC configuration to evaluate.
        """


class MakeInitialState(Protocol):

    def __call__(self,
               atoms: jnp.ndarray,
               charges: jnp.ndarray,
               nspins: Sequence[int],
               use_scan: bool = False,
               **kwargs: Any) -> InitialState:
        """Builds the InitialState function.

        Args:
          **kwargs: additional kwargs to use for creating the specific Hamiltonian.
        """

    
def initial_state(atoms: jnp.ndarray,
                  charges: jnp.ndarray,
                  nspins: Sequence[int],
                  use_scan: bool = False) -> InitialState:
    """Creates the function to evaluate the local energy."""

    def _i_s(data: jnp.ndarray) -> jnp.ndarray:
        
        ae, _, r_ae, _ = networks.construct_input_features(data, atoms)
        
        m, n, _ = r_ae.shape

        pro_dis = jnp.zeros((m,n,1))
        print(pro_dis)
        for i in range(m):
            for j in range(n):
                if r_ae[i,j,1] <= 1:
                    pro_dis = pro_dis.at[i,j,0].set(3/(4 * jnp.pi))

        return pro_dis

    return _i_s


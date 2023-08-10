from typing import Any, Sequence

from ferminet import networks
import jax
from jax import lax
import jax.numpy as jnp
from typing_extensions import Protocol
from jax.scipy.stats import multivariate_normal


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

        m = data.size
        mean = jnp.zeros(m)
        cov = 5 * jnp.eye(m)
        
        pro_dis = multivariate_normal.pdf(data, mean, cov)
        pro_dis_sq = jnp.sprt(pro_dis)

        return pro_dis_sq

    return _i_s
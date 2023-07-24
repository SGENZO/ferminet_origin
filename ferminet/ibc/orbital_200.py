import jax
from jax import numpy as jnp
from jax.scipy.special import lpmn_values
from jax.scipy.special import sph_harm
from typing import Any, Sequence
from ferminet import networks
from jax import lax
from scipy.special import assoc_laguerre
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

        x = data[0]
        y = data[1]
        z = data[2]

        r = (x**2 + y**2 + z**2)**0.5
        theta = jnp.arccos(z/r)
        phi = jnp.arctan(y/x)

        n = 2
        l = 0
        m = 0

        
        rad = jnp.exp(-r/n) # * assoc_laguerre(2*r/n, n-l-1, 2*l+1)
        sph = sph_harm(m, l , theta, phi)
        result = rad * sph

        return result

    return _i_s
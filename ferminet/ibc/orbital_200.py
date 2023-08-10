import jax
from jax import numpy as jnp
from scipy.special import sph_harm
from scipy.special import assoc_laguerre
from typing import Any, Sequence
from ferminet import networks
from jax import lax
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

        #data_reshape = data.reshape(-1,3)

        #x = data_reshape[0,:].reshape(-1)
        #y = data_reshape[1,:].reshape(-1)
        #z = data_reshape[2,:].reshape(-1)
        # lenx = x.size
        x = data[0]
        y = data[1]
        z = data[2]

        r = (x**2 + y**2 + z**2)**0.5
        theta = jnp.arccos(z/r)
        phi = jnp.arctan(y/x)
        theta = jnp.array(theta)
        phi = jnp.array(phi)
        #print(r)

        n = 2 #* jnp.ones(lenx)
        l = 0 #* jnp.ones(lenx)
        m = 0 #* jnp.ones(lenx)

        Sph = sph_harm(m, l, theta, phi)
        rho = r / n
        Lag = assoc_laguerre(2 * rho, n - l - 1, 2 * l + 1)
        result = jnp.array(jnp.exp(-rho) * jnp.power((2*rho),l) * Lag * Sph)


        return result

    return _i_s
import jax
from jax import numpy as jnp
from jax.scipy.special import lpmn_values
from jax.scipy.special import sph_harm

def make_orbital(n,l,m):
    def orbital(r, theta, phi):
        
        rad = jnp.exp(-r/n) * lpmn_values(2*l+1, n-l-1, 2*r/n, is_normalized=Ture)
        sph = sph_harm(m, l , theta, phi)
        result = rad * sph

        return result 

    return orbital
        
    
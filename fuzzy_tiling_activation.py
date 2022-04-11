from typing import Union

import chex
import jax.numpy as jnp
from jax import nn
import haiku as hk


class FuzzyTilingActivation(hk.Module):
    """Fuzzy Tiling Activations:
    A Simple Approach to Learning Sparse Representations Online

    2019, Yangchen Pan, Kirby Banman, Martha White
    https://arxiv.org/abs/1911.08068

    If eta is None it defaults to (high - low) / n_tiles.

    If layer_norm_input is True, input is normalized before tiling and low and
    high are set to -3 and 3 respectively. If sigmoid_input is True, input sent
    through sigmoid before tiling and low and high are set to 0 and 1
    respectively. Both can be True in which case sigmoid is after layer norm and
    low, high = 0, 1.
    """

    def __init__(
        self,
        n_tiles: int = 20,
        low: float = -2.0,
        high: float = 2.0,
        eta: Union[float, None] = None,
        layer_norm_input: bool = False,
        sigmoid_input: bool = False,
    ):
        """

        :param n_tiles: Number of bins per scalar.
        :param low: low of input range.
        :param high: high of input range.
        :param eta: If None, defaults to (high - low) / n_tiles.
        :param layer_norm_input: If True, input is normalized before tiling and
            low and high are set to -3 and 3 respectively.
        :param sigmoid_input: If True, input sent through sigmoid before tiling
            and low and high are set to 0 and 1 respectively.
        """
        super().__init__()
        if layer_norm_input:
            low = -3.0
            high = 3.0
            if eta is not None:
                print(
                    "WARNING: Double check eta value when using layer_norm_input "
                    "as low and high become -3 and 3 respectively. Set eta to "
                    "None to use default value (high - low) / n_tiles."
                )
        if sigmoid_input:
            low = 0.0
            high = 1.0
            if eta is not None:
                print(
                    "WARNING: Double check eta value when using sigmoid_input "
                    "as low and high become 0 and 1 respectively. Set eta to "
                    "None to use default value (high - low) / n_tiles."
                )
        self.k = n_tiles
        self.low = low
        self.high = high
        self.layer_norm_input = layer_norm_input
        self.sigmoid_input = sigmoid_input
        if eta:
            self.eta = eta
        else:
            self.eta = (high - low) / n_tiles
        self.delta = (high - low) / n_tiles
        self.c_vec = jnp.arange(n_tiles, dtype=jnp.float32) * self.delta + low

    def __call__(self, x: chex.Array) -> chex.Array:
        if self.layer_norm_input:
            x = hk.LayerNorm(-1, True, True)(x)
        if self.sigmoid_input:
            x = nn.sigmoid(x)
        d = x.shape[-1]
        x = jnp.expand_dims(x, -1)
        sum_relu = nn.relu(self.c_vec - x) + nn.relu(x - self.delta - self.c_vec)
        return jnp.reshape(
            (1.0 - i_plus_eta(sum_relu, self.eta)),
            [-1, d * self.k],
        )


def i_plus_eta(x: chex.Array, eta: float) -> chex.Array:
    if eta == 0.0:
        return jnp.sign(x)
    return jnp.less_equal(x, eta).astype(jnp.float32) * x + jnp.greater(x, eta).astype(
        jnp.float32
    )

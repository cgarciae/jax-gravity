import jax.numpy as jnp
import jax
import typing as tp

F = tp.TypeVar("F", bound=tp.Callable[..., tp.Any])


def safe_norm(a, b) -> jax.Array:
    r = b - a
    r = jnp.where(jnp.allclose(b - a, 0.0), 1.0, r)
    r = jnp.linalg.norm(r, ord=2, axis=-1)
    r = jnp.where(jnp.allclose(b - a, 0.0), 1e-8, r)
    return r


def map_product(f: F) -> F:
    return jax.vmap(jax.vmap(f, (0, None), 0), (None, 0), 1)


def mask_diagonal(x) -> jax.Array:
    return jnp.where(jnp.eye(x.shape[-1]), 0.0, x)

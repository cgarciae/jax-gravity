"""
In this script we are going to create a gravitational simulation of N bodies using JAX.
We will use the jax.experimental.ode package to solve the ODE system.
"""
from functools import partial

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax.experimental import ode
from matplotlib import animation
import numpy as np

# set jax to double precision
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_debug_nans", True)

# RNG keys
seed = jax.random.PRNGKey(42)
kx, kv, km = jax.random.split(seed, 3)

# Simulation Parameters
N_BODIES = 3
T0 = 0.0
months = 15
Tf = 3600.0 * 30.5 * months  # in seconds
DT = 1.0  # 1 second
T = jnp.arange(T0, Tf, DT)
# T = jnp.arange(T0, T0 + DT * 10000, DT)
G = jnp.array(6.67408e-11, dtype=jnp.float32)

# Initial Values
X0 = jax.random.uniform(kx, shape=(N_BODIES, 3), minval=-748e7, maxval=748e7)
V0 = jax.random.uniform(kv, shape=(N_BODIES, 3), minval=-1e5, maxval=1e5)
M = jax.random.uniform(km, shape=(N_BODIES, 1), minval=5.972e27, maxval=1.898e30)
M = M[:, 0]
# ignore z-axis
X0 = X0.at[:, 2].set(0.0)
V0 = V0.at[:, 2].set(0.0)

# system state
Y = (X0, V0)


# -----------------------------------------------
# Gravity
# -----------------------------------------------
# Here we calculate the interation between two bodies only
# but use vmap twice automatically perform broadcasting for us:
# we alternate which bodies bodies we are slicing and which we are replicating
# to construct the full matrix of interactions.
def gravitational_energy(
    a: tuple[jax.Array, jax.Array],
    b: tuple[jax.Array, jax.Array],
) -> jax.Array:
    xa, ma = a
    xb, mb = b
    r = safe_norm(xa, xb)
    energy = -G * mb / r
    energy = jnp.where(jnp.allclose(xa, xb), 0.0, energy)
    return energy


# -----------------------------------------------
# System dynamics
# -----------------------------------------------
# Here we use the gravity function to calculate calculated the acceleration
# which is the derivative of the velocity (dV), the derivative of the position
# is just the velocity (dX). Notice how the state is conveniently a pytree tuple :)


def force_fn(a: tuple[jax.Array, jax.Array]) -> jax.Array:
    def total_energey(a, b):
        return -0.5 * map_product(gravitational_energy)(a, b).sum()

    return jax.grad(total_energey)(a, a)[0]


def force_fn2(a: tuple[jax.Array, jax.Array]) -> jax.Array:
    force = -map_product(jax.grad(gravitational_energy))(a, a)[0]
    return 2 * mask_diagonal(force).sum(axis=1)


def dY(Y, t):
    X, V = Y

    # F =
    F = force_fn2((X, M))
    dV = F
    dX = V
    return dX, dV


# -----------------------------------------------
# ODE Solver
# -----------------------------------------------
# We call `odeint` which solves the system of differential equations
# using the 4th order Runge-Kutta method. The returned X and V values
# have a the shape (time, bodies, dimensions)
(X, V) = ode.odeint(dY, Y, T)
X, V = np.asarray(X), np.asarray(V)


# -----------------------------------------------
# Create animation
# -----------------------------------------------
# Here we use the animation module from matplotlib to create an animation
# of the system. The animate function iteratively updates some lines and scatter
# plots.

fig, ax = plt.subplots()

ax.axis("off")
ax.set_xlim(X[..., 0].min() * 1.3, X[..., 0].max() * 1.3)
ax.set_ylim(X[..., 1].min() * 1.3, X[..., 1].max() * 1.3)

lines = tuple([ax.plot(X[:0, j, 0], X[:0, j, 1])[0] for j in range(N_BODIES)])
scatter = ax.scatter(X[0, :, 0], X[0, :, 1])


def animate(i):
    i0 = max(0, i - 100_000)
    for j in range(N_BODIES):
        lines[j].set_data(X[i0:i, j, 0], X[i0:i, j, 1])

    scatter.set_offsets(X[i, :, :2])

    return lines + (scatter,)


anim = animation.FuncAnimation(
    fig,
    animate,
    init_func=lambda: animate(0),
    frames=range(2, len(X), 1000),
    interval=20,
    blit=True,
)


plt.show()

print("Saving animation...")
# anim.save("animation2.mp4", writer="ffmpeg")

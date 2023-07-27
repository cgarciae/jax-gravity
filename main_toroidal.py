"""
In this script we are going to create a gravitational simulation of N bodies using JAX.
We will use the jax.experimental.ode package to solve the ODE system.
"""
from functools import partial
from math import exp
from typing import Any, TypeVar

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax.experimental import ode
from matplotlib import animation, rc
from diffrax import ODETerm, Tsit5
import numpy as np

# set jax to double precision
jax.config.update("jax_enable_x64", True)

# type aliases
A = TypeVar("A")

# RNG keys
seed = jax.random.PRNGKey(42)
kx, kv, km = jax.random.split(seed, 3)

# Simulation Parameters
N_BODIES = 3
T0 = 0.0
months = 12
Tf = 3600.0 * 30.5 * months  # in seconds
DT = 1.0  # 1 second
T = jnp.arange(T0, Tf, DT)
G = jnp.array(6.67408e-11, dtype=jnp.float32)
L: float = 4 * 748e7

# Initial Values
X0 = jax.random.uniform(kx, shape=(N_BODIES, 3), minval=-L / 2, maxval=L / 2)
V0 = jax.random.uniform(kv, shape=(N_BODIES, 3), minval=-1e5, maxval=1e5)
M = jax.random.uniform(km, shape=(N_BODIES, 1), minval=5.972e27, maxval=1.898e30)
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
@partial(jax.vmap, in_axes=(0, 0, None, None))
@partial(jax.vmap, in_axes=(None, None, 0, 0))
def gravity(
    xa: jnp.ndarray,
    ma: jnp.ndarray,
    xb: jnp.ndarray,
    mb: jnp.ndarray,
) -> jnp.ndarray:
    radius3 = jnp.linalg.norm(xb - xa) ** 3
    f = G * ma * mb * (xb - xa) / radius3

    # fill nan values with 0s
    return jnp.nan_to_num(f)


# -----------------------------------------------
# Toroidal Topology
# -----------------------------------------------
def virtual_particles(X, M):
    right = X + jnp.array([L, 0.0, 0.0])
    right_up = X + jnp.array([L, L, 0.0])
    up = X + jnp.array([0.0, L, 0.0])
    left_up = X + jnp.array([-L, L, 0.0])
    left = X + jnp.array([-L, 0.0, 0.0])
    left_down = X + jnp.array([-L, -L, 0.0])
    down = X + jnp.array([0.0, -L, 0.0])
    right_down = X + jnp.array([L, -L, 0.0])

    X = jnp.concatenate(
        [X, right, right_up, up, left_up, left, left_down, down, right_down]
    )
    M = jnp.concatenate([M] * 9)

    return X, M


def boundary_conditions(Y) -> Any:
    X, V = Y
    X = jnp.mod(X, L)
    return X, V


def gravitation_field(x, m, n):
    m0 = m
    x = x[None]
    m = m[None]

    xs, ys = jnp.linspace(0, L, n), jnp.linspace(0, L, n)
    xx, yy = jnp.meshgrid(xs, ys)
    Xgrid = jnp.stack([xx, yy, jnp.zeros_like(xx)], axis=-1).reshape(-1, 3)
    Mgrid = jnp.full(Xgrid.shape[0:1], m0)

    x, m = virtual_particles(x, m)

    F = gravity(Xgrid, Mgrid, x, m).sum(axis=1)
    # add field from virtual particles
    # F = F.reshape(-1, 9, 3).sum(axis=1)
    # remove z-component
    F = F[:, :2]

    return xx, yy, F.reshape(*xx.shape, 2)


# plot gravitational field
x_test = jnp.array([L / 4, L / 4, 0.0])
m_test = jnp.array([1e30])

xx, yy, F = gravitation_field(x_test, m_test, n=61)

# quiver plot
fig, ax = plt.subplots(figsize=(10, 10))
exp = 1.1
Fnorm = F / jnp.linalg.norm(F, axis=-1, ord=2, keepdims=True)
F = np.sign(F) * (np.abs(F) ** 0.2)
ax.quiver(xx, yy, F[..., 0], F[..., 1])
ax.set_xlim(0, L)
ax.set_ylim(0, L)
plt.show()

exit()


# -----------------------------------------------
# System dynamics
# -----------------------------------------------
# Here we use the gravity function to calculate calculated the acceleration
# which is the derivative of the velocity (dV), the derivative of the position
# is just the velocity (dX). Notice how the state is conveniently a pytree tuple :)
def dY(t, Y, args):
    X, V = Y
    Xv, Mv = virtual_particles(X, M)
    F = gravity(X, M, Xv, Mv).sum(axis=1)
    dV = F / M
    dX = V
    return dX, dV


# -----------------------------------------------
# ODE Solver
# -----------------------------------------------
# We call `odeint` which solves the system of differential equations
# using the 4th order Runge-Kutta method. The returned X and V values
# have a the shape (time, bodies, dimensions)


def odeint(f, y0: A, t) -> A:
    term = ODETerm(f)
    solver = Tsit5()

    tprev = t[0]
    tnext = t[1]
    y0 = boundary_conditions(y0)
    args = None
    state = solver.init(term, tprev, tnext, y0, args)

    def scan_fn(carry, tnext):
        y, state, tprev = carry
        y, _, _, state, _ = solver.step(
            term, tprev, tnext, y, args, state, made_jump=False
        )
        y = boundary_conditions(y)
        return (y, state, tnext), y

    _, y = jax.lax.scan(scan_fn, (y0, state, tprev), t[1:])

    y = jax.tree_map(lambda y0, y: jnp.concatenate([y0[None, ...], y], axis=0), y0, y)

    return y


(X, V) = odeint(dY, Y, T)
X, V = np.asarray(X), np.asarray(V)

# -----------------------------------------------
# Create animation
# -----------------------------------------------
# Here we use the animation module from matplotlib to create an animation
# of the system. The animate function iteratively updates some lines and scatter
# plots.

fig, ax = plt.subplots()

ax.axis("off")
ax.set_xlim(0, L)
ax.set_ylim(0, L)

lines = tuple([ax.plot(X[:0, j, 0], X[:0, j, 1])[0] for j in range(N_BODIES)])
scatter = ax.scatter(X[0, :, 0], X[0, :, 1])


def animate(i):
    i0 = max(0, i - 100_000)
    for j in range(N_BODIES):
        Xj = X[i0:i, j]
        d = np.linalg.norm(Xj[:-1] - Xj[1:], axis=1)
        Xj = np.where(d[:, None] > L / 2, np.nan, Xj[:-1])
        lines[j].set_data(Xj[:, 0], Xj[:, 1])

    scatter.set_offsets(X[i, :, :2])

    return lines + (scatter,)


anim = animation.FuncAnimation(
    fig,
    animate,
    init_func=lambda: animate(0),
    frames=range(2, len(X), 500),
    interval=20,
    blit=True,
)


plt.show()

print("Saving animation...")
anim.save("animation.gif", writer="imagemagick")

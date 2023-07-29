"""
In this script we are going to create a gravitational simulation of N bodies using JAX.
We will use the jax.experimental.ode package to solve the ODE system.
"""
import json
from pathlib import Path
import time
import typing as tp

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib import animation
from diffrax import ODETerm, Tsit5
import numpy as np
import utils

# set jax to double precision
jax.config.update("jax_enable_x64", True)

# type aliases
A = tp.TypeVar("A")


class Body(tp.NamedTuple):
    x: jax.Array
    v: jax.Array
    m: jax.Array


# RNG keys
seed = 42
key = jax.random.PRNGKey(seed)
kx, kv, km = jax.random.split(key, 3)

# Simulation Parameters
N_BODIES = 5
T0 = 0.0
months = 30
Tf = 3600.0 * 30.5 * months  # in seconds
DT = 1.0  # 1 second
T = jnp.arange(T0, Tf, DT)
G = jnp.array(6.67408e-11, dtype=jnp.float32)
L: float = 8 * 748e7
grid_radius = 3
plot_field = False
toroid_rstride = 5
toroid_cstride = 5
initial_speed = 1e5
max_speed = 1e7
max_force = 1e35

# Initial Values
X0 = jax.random.uniform(kx, shape=(N_BODIES, 3), minval=L / 4, maxval=L * 3 / 4)
V0 = jax.random.uniform(
    kv, shape=(N_BODIES, 3), minval=-initial_speed, maxval=initial_speed
)
M = jax.random.uniform(km, shape=(N_BODIES, 1), minval=5.972e27, maxval=1.898e30)
# ignore z-axis
X0 = X0.at[:, 2].set(0.0)
V0 = V0.at[:, 2].set(0.0)
date = str(int(time.time()))

# system state
Y = Body(X0, V0, M)

# -----------------------------------------------
# Utils
# -----------------------------------------------


def clip_by_norm(x, norm):
    x_norm = jnp.linalg.norm(x, axis=-1, ord=2, keepdims=True)
    clipped_x = jnp.where(x_norm > norm, x / x_norm * norm, x)
    return clipped_x


def polar_to_toroid(theta, phi):
    c, a = 2, 1
    x = (c + a * np.cos(theta)) * np.cos(phi)
    y = (c + a * np.cos(theta)) * np.sin(phi)
    z = a * np.sin(theta)

    return x, y, z


def cartesian_to_polar(x, y):
    theta = y / L * 2 * np.pi + np.pi
    phi = x / L * 2 * np.pi + np.pi
    return theta, phi


# -----------------------------------------------
# Gravity
# -----------------------------------------------
# Here we calculate the interation between two bodies only
# but use vmap twice automatically perform broadcasting for us:
# we alternate which bodies bodies we are slicing and which we are replicating
# to construct the full matrix of interactions.
def gravitational_force(a: Body, b: Body) -> jax.Array:
    radius3 = utils.safe_norm(b.x, a.x) ** 3
    f = G * b.m * (b.x - a.x) / radius3
    # fill nan values with 0s
    return jnp.nan_to_num(f)


def gravitational_energy(a: Body, b: Body) -> jax.Array:
    r = utils.safe_norm(a.x, b.x)
    energy = -G * b.m / r
    energy = jnp.where(jnp.allclose(a.x, b.x), 0.0, energy)
    return energy


# -----------------------------------------------
# Toroidal Topology
# -----------------------------------------------
def virtual_particles(a: Body, R: int) -> Body:
    space = jnp.linspace(-L * R, L * R, 2 * R + 1)
    xx, yy = jnp.meshgrid(space, space)
    zz = jnp.zeros_like(xx)
    Xgrid = jnp.stack([xx, yy, zz], axis=-1).reshape(-1, 3)

    x = a.x[None] + Xgrid[:, None]
    x = x.reshape(-1, 3)

    m = jnp.concatenate([a.m] * (2 * R + 1) ** 2)
    v = jnp.concatenate([a.v] * (2 * R + 1) ** 2)

    return Body(x=x, v=v, m=m)


def boundary_conditions(a: Body) -> Body:
    x = jnp.mod(a.x, L)  # toroidal topology
    v = clip_by_norm(a.v, max_speed)  # clip velocity
    return Body(x=x, v=v, m=a.m)


def gravitation_field(x, m, n, grid_radius):
    m0 = m
    x = x[None]
    m = m[None]

    xs, ys = jnp.linspace(0, L, n), jnp.linspace(0, L, n)
    xx, yy = jnp.meshgrid(xs, ys)
    Xgrid = jnp.stack([xx, yy, jnp.zeros_like(xx)], axis=-1).reshape(-1, 3)
    Mgrid = jnp.full(Xgrid.shape[0:1], m0)
    grid = Body(x=Xgrid, v=jnp.zeros_like(Xgrid), m=Mgrid)

    a = Body(x=x, v=jnp.zeros_like(x), m=m)
    virtual = virtual_particles(a, grid_radius)

    force = gravitational_force(grid, virtual).sum(axis=1)
    # add field from virtual particles
    # F = F.reshape(-1, 9, 3).sum(axis=1)
    # remove z-component
    force = force[:, :2]

    return xx, yy, force.reshape(*xx.shape, 2)


# plot gravitational field
if plot_field:
    x_test = jnp.array([L / 4, L / 4, 0.0])
    m_test = jnp.array([1e30])

    xx, yy, F = gravitation_field(x_test, m_test, n=45, grid_radius=200)

    fig, ax = plt.subplots(figsize=(10, 10))
    # plot stable points
    plt.scatter(
        [L / 4, L / 4, L * 3 / 4, L * 3 / 4],
        [L / 4, L * 3 / 4, L / 4, L * 3 / 4],
        c="b",
    )
    # plot field
    Fnorm = F / jnp.linalg.norm(F, axis=-1, ord=2, keepdims=True)
    F = np.sign(F) * (np.abs(F) ** 0.22)
    ax.quiver(xx, yy, F[..., 0], F[..., 1], minlength=0.2, width=0.0015)
    ax.set_xlim(0, L)
    ax.set_ylim(0, L)

    fig.savefig("toroidal_gravitation_field.png")


# -----------------------------------------------
# System dynamics
# -----------------------------------------------
# Here we use the gravity function to calculate calculated the acceleration
# which is the derivative of the velocity (dV), the derivative of the position
# is just the velocity (dX). Notice how the state is conveniently a pytree tuple :)
def dY(t, y: Body, args):
    virtual = virtual_particles(y, grid_radius)
    force = utils.map_product(gravitational_force)(y, virtual).sum(axis=1)
    dV = force
    dX = y.v
    return Body(x=dX, v=dV, m=jnp.zeros_like(y.m))


# -----------------------------------------------
# ODE Solver
# -----------------------------------------------
# We call `odeint` which solves the system of differential equations
# using the 4th order Runge-Kutta method. The returned X and V values
# have a the shape (time, bodies, dimensions)


def odeint(f, y0: Body, t) -> Body:
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


Y = odeint(dY, Y, T)
Y = jax.tree_map(np.asarray, Y)
X = Y.x

# -----------------------------------------------
# Create animation
# -----------------------------------------------
# Here we use the animation module from matplotlib to create an animation
# of the system. The animate function iteratively updates some lines and scatter
# plots.


def create_animation(
    projection: bool, cartesian: bool, format: tp.Optional[str], show: bool
):
    assert projection or cartesian

    h = 20 if projection and cartesian else 10
    fig = plt.figure(figsize=(10, h))

    ## Plot toroidal topology
    if projection:
        n = 50
        theta = np.linspace(0, 2.0 * np.pi, n)
        phi = np.linspace(0, 2.0 * np.pi, n)
        theta, phi = np.meshgrid(theta, phi)

        x, y, z = polar_to_toroid(theta, phi)

        subplot_pos = 211 if cartesian else 111
        ax_toroid = fig.add_subplot(subplot_pos, projection="3d")
        ax_toroid.set_zlim(-1, 1)
        ax_toroid.set_xlim(-3, 3)
        ax_toroid.plot_surface(
            x,
            y,
            z,
            rstride=toroid_rstride,
            cstride=toroid_cstride,
            color="w",
            edgecolors="k",
            alpha=0,
            linewidths=0.5,
            linestyle=":",
        )
        ax_toroid.view_init(36, 26)
        ax_toroid.axis("off")

        scatter_toroid = ax_toroid.scatter(
            *polar_to_toroid(*cartesian_to_polar(X[0, :, 0], X[0, :, 1]))
        )

        lines_toroid = tuple(
            ax_toroid.plot(
                *polar_to_toroid(*cartesian_to_polar(X[:0, j, 0], X[:0, j, 1]))
            )[0]
            for j in range(N_BODIES)
        )
    if cartesian:
        ## Plot Cartesian
        subplot_pos = 212 if projection else 111
        ax_cartesian = fig.add_subplot(subplot_pos)
        ax_cartesian.axis("off")
        ax_cartesian.set_xlim(0, L)
        ax_cartesian.set_ylim(0, L)

        lines_cartesian = tuple(
            ax_cartesian.plot(X[:0, j, 0], X[:0, j, 1])[0] for j in range(N_BODIES)
        )
        scatter_cartesian = ax_cartesian.scatter(X[0, :, 0], X[0, :, 1])

    def animate(i):
        i0 = max(0, i - 100_000)
        for j in range(N_BODIES):
            Xj = X[i0:i, j]
            d = np.linalg.norm(Xj[:-1] - Xj[1:], axis=1)
            Xj = np.where(d[:, None] > L / 2, np.nan, Xj[:-1])

            if cartesian:
                lines_cartesian[j].set_data(Xj[:, 0], Xj[:, 1])
            if projection:
                lines_toroid[j]._verts3d = polar_to_toroid(
                    *cartesian_to_polar(Xj[:, 0], Xj[:, 1])
                )

        if cartesian:
            scatter_cartesian.set_offsets(X[i, :, :2])
        if projection:
            scatter_toroid._offsets3d = polar_to_toroid(
                *cartesian_to_polar(X[i, :, 0], X[i, :, 1])
            )

        plots = ()

        if cartesian:
            plots += lines_cartesian + (scatter_cartesian,)
        if projection:
            plots += lines_toroid + (scatter_toroid,)

        return plots

    anim = animation.FuncAnimation(
        fig,
        animate,
        init_func=lambda: animate(0),
        frames=range(2, len(X), 1000),
        interval=20,
        blit=False,
    )

    if show:
        plt.show()

    if format is not None:
        if projection and cartesian:
            name = "projection_and_cartesian"
        elif projection:
            name = "projection"
        else:
            name = "cartesian"

        name = f"{path}/toroidal_{name}.{format}"

        print(f"Saving animation '{name}'...")
        anim.save(name, writer="ffmpeg")


path = Path("output") / date
path.mkdir(parents=True, exist_ok=True)

json.dump(
    dict(
        grid_radius=grid_radius,
        months=months,
        seed=seed,
        toroid_rstride=toroid_rstride,
        toroid_cstride=toroid_cstride,
        initial_speed=initial_speed,
        L=L,
        N_BODIES=N_BODIES,
        max_speed=max_speed,
        max_force=max_force,
    ),
    (path / "config.json").open("w"),
)

create_animation(projection=True, cartesian=False, format=None, show=False)
# create_animation(projection=True, cartesian=False, format="mp4", show=False)
# create_animation(projection=True, cartesian=True, format="mp4", show=False)
# create_animation(projection=False, cartesian=True, format="mp4", show=False)
